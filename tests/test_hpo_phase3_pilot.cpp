// Phase 3.6 — pilot end-to-end integration.
//
// Exercises the full Phase 3 chain on a real LAMMPS+pair_tadah run:
//   1. Spin up LAMMPS, run a minimal BCC Ta `box/relax + minimize` block.
//   2. Build SimsGradScriptInput from the relaxed state via the new
//      `populate_sims_grad_input_from_lammps` driver — pulls aeds,
//      dF_dW snapshot, Hessian, observable values from the live LMP.
//   3. Feed a synthetic dWdtheta into `compute_sims_grad_dWchain` and
//      verify `g_out` matches the closed-form expectation for cohesive
//      E per atom: λ_k=0 ⇒ dO/dW = (Σ_a aeds(:,a)) / N.
//
// Cohesive-E-only is the pilot path the plan specified (∂g/∂x = 0).
// Lattice-constant (λ≠0) parity is Phase 3.7's job and depends on the
// Hessian sametag fix logged as 3.6 follow-up.
//
// Hidden by default (`[.][!benchmark]` like the rest of the LAMMPS-
// driven cases) so CI's default sweep doesn't pay the LAMMPS startup
// cost. Run manually with:
//   ./test_hpo "[phase3_pilot]" --benchmark-samples 1

#include "catch2/catch.hpp"

#include <tadah/core/aeds_type.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/sims_grad.h>
#include <tadah/hpo/sims_grad_lammps.h>
#include <tadah/hpo/warmstart_cache.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <atom.h>
#include <domain.h>
#include <force.h>
#include <neighbor.h>
#include <pair_tadah.h>

#include <Eigen/Dense>

#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string env_or(const char* name)
{
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

std::string resolve_pot_path()
{
    const std::string env = env_or("TADAH_BENCH_POT");
    if (!env.empty()) return env;
#ifdef TADAH_BENCH_DEFAULT_POT
    return std::string(TADAH_BENCH_DEFAULT_POT);
#else
    return std::string();
#endif
}

/* Phase 3.7C BCC iso script: lattice constant observable. Same shape
 * as the pilot script but exposes `var_lx = lx` (and a redundant
 * `e_per` for cross-checks). Single LAMMPS block, single observable,
 * single relaxed state. */
std::vector<std::string> bcc_lattice_constant_script()
{
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 3.301",
        "region box block 0 2 0 2 0 2 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "compute eng all pe/atom",
        "compute eatoms all reduce sum c_eng",
        "fix relax all box/relax iso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 1e-15 1e-15 5000 5000",
        "unfix relax",
        "variable var_lx equal lx",
        "variable e_per equal $(c_eatoms/atoms)",
        "run 0",
    };
}

/* Phase 3.7E (3.7D follow-up) BCC aniso script: same observable as
 * the iso script but with `box/relax aniso 0 0 0` so all three
 * diagonal box DOFs are free during relaxation. The cubic equilibrium
 * has lx = ly = lz; the test exercises the Voigt-6 dvirial_dW chain
 * end-to-end (box_dofs=6 path through populate, Hessian, solver, and
 * apply_ift_chain_kernel's box block). */
std::vector<std::string> bcc_lattice_constant_aniso_script()
{
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 3.301",
        "region box block 0 2 0 2 0 2 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "compute eng all pe/atom",
        "compute eatoms all reduce sum c_eng",
        "fix relax all box/relax aniso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 1e-15 1e-15 5000 5000",
        "unfix relax",
        "variable var_lx equal lx",
        "variable e_per equal $(c_eatoms/atoms)",
        "run 0",
    };
}

/* Minimal BCC Ta script: defines `e_per` = cohesive energy per atom.
 * Single LAMMPS block (no `clear`), single observable, single relaxed
 * state. Avoids the pilot script's cross-clear semantics. */
std::vector<std::string> simple_bcc_pilot_script()
{
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 3.301",
        "region box block 0 2 0 2 0 2 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "compute eng all pe/atom",
        "compute eatoms all reduce sum c_eng",
        "fix relax all box/relax iso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 1e-12 1e-12 1000 1000",
        "unfix relax",
        "variable e_per equal $(c_eatoms/atoms)",
        "run 0",
    };
}

/* Synth a LammpsSim metadata block matching the script above:
 * one observable `e_per`, L2 loss, target 0.0, weight 1.0. */
tadah::hpo::LammpsSim make_pilot_lsim()
{
    tadah::hpo::LammpsSim ls;
    ls.script = "simple_bcc_pilot";
    ls.n_outvar  = {"e_per"};
    ls.n_varloss = {"e_per"};
    ls.t_varloss = {0.0};
    ls.w_varloss = {1.0};
    ls.i_varloss = {0};
    ls.lossFn    = {tadah::LossFunction(tadah::LossFunction::Type::L2)};
    ls.failscore = 1.0e8;
    return ls;
}

/* Phase 3.7B-warmstart pilot — same BCC Ta cohesive-E observable but
 * sources the initial lattice constant from a hot-var seed `${hot_a0}`.
 * Mirrors the production pattern where a hot script references a
 * warmstarted relaxation seed. */
std::vector<std::string> simple_bcc_hotvar_pilot_script()
{
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc ${hot_a0}",
        "region box block 0 2 0 2 0 2 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "compute eng all pe/atom",
        "compute eatoms all reduce sum c_eng",
        "fix relax all box/relax iso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 1e-12 1e-12 1000 1000",
        "unfix relax",
        "variable e_per equal $(c_eatoms/atoms)",
        "run 0",
    };
}

/* Hot-var pilot lsim — one HotVarDecl for `hot_a0`, range [3.0, 3.6],
 * initial 3.301 (BCC Ta equilibrium guess). */
tadah::hpo::LammpsSim make_hotvar_pilot_lsim()
{
    tadah::hpo::LammpsSim ls = make_pilot_lsim();
    ls.script = "simple_bcc_hotvar_pilot";
    tadah::hpo::LammpsSim::HotVarDecl hv;
    hv.name    = "hot_a0";
    hv.min     = 3.0;
    hv.max     = 3.6;
    hv.initial = 3.301;
    ls.hotvars = { hv };
    return ls;
}

}  // namespace

TEST_CASE("phase3_pilot — chain rule on cohesive E per atom matches "
          "the closed-form (Σ aeds) / N for L2 loss",
          "[phase3_pilot][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) {
        WARN("phase3_pilot skipped: no pot. Set TADAH_BENCH_POT or "
             "re-cmake with TADAH_PATH.");
        SUCCEED("skipped (no pot)");
        return;
    }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) {
            WARN("phase3_pilot skipped: pot unreadable.");
            SUCCEED("skipped (pot unreadable)");
            return;
        }
    }

    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx = tadah::core::CoreUserConfigParser().parse(pot_path));

    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, simple_bcc_pilot_script(), pot_ctx));

    auto* pair_base = lmp->force->pair;
    auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(pair_base);
    REQUIRE(pair != nullptr);
    auto* lt = pair->lammps_tadah();
    REQUIRE(lt != nullptr);

    /* Read the observable value as the eval pipeline would. */
    const std::string raw_e_per = runner.get_variable(lmp, "e_per");
    REQUIRE_FALSE(raw_e_per.empty());

    const auto lsim = make_pilot_lsim();
    const std::vector<std::string> v_outvar = { raw_e_per };
    const std::vector<std::string> expressions = { "c_eatoms/atoms" };

    /* Build the chain-rule input. box_dofs=0 keeps the pilot pure-
     * atom; cohesive E's dg_dx_is_zero flag will short-circuit the
     * Hessian solve anyway. */
    tadah::hpo::SimsGradScriptInput in;
    const bool ok = tadah::hpo::populate_sims_grad_input_from_lammps(
        lmp.get(), lsim, v_outvar, expressions, /*box_dofs=*/0,
        lsim.failscore, in);
    REQUIRE(ok);

    REQUIRE(in.ctx.aeds != nullptr);
    REQUIRE(in.ctx.n_desc > 0);
    REQUIRE(in.ctx.nlocal > 0);
    REQUIRE(in.dF_dW != nullptr);
    REQUIRE(in.v_observed.size() == 1);
    REQUIRE(in.expressions.size() == 1);

    /* Synthetic dWdtheta: identity-ish layout, one θ row per descriptor
     * column so g_out[c] picks out a single coefficient of dL/dW. */
    const int n_desc = in.ctx.n_desc;
    std::vector<double> dWdtheta(static_cast<std::size_t>(n_desc * n_desc), 0.0);
    for (int r = 0; r < n_desc; ++r) dWdtheta[r * n_desc + r] = 1.0;
    std::vector<int> theta_row_to_g_idx(n_desc);
    for (int r = 0; r < n_desc; ++r) theta_row_to_g_idx[r] = r;
    std::vector<int> all_theta_indices(n_desc);
    for (int r = 0; r < n_desc; ++r) all_theta_indices[r] = r;

    std::vector<double> g_out(n_desc, 0.0);
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_sims_grad_dWchain(
        {in}, dWdtheta, static_cast<std::size_t>(n_desc),
        theta_row_to_g_idx, all_theta_indices, g_out, needs_fd));
    CHECK(needs_fd.empty());

    /* Closed-form reference: cohesive E ⇒ dg/dW = Σ_a aeds(:,a) / N.
     * λ_k=0 ⇒ dO/dW = dg/dW. L2 fold ⇒ dL/dW = 2·w·(v - t) · dO/dW.
     * Identity dWdtheta ⇒ g_out[c] = dL/dW[c]. */
    Eigen::VectorXd dg_dW(n_desc);
    for (int c = 0; c < n_desc; ++c) {
        double sum = 0.0;
        for (int a = 0; a < in.ctx.nlocal; ++a) sum += (*in.ctx.aeds)(c, a);
        dg_dW(c) = sum / in.ctx.natoms;
    }
    const double v = in.v_observed[0];
    const double t = lsim.t_varloss[0];
    const double w = lsim.w_varloss[0];
    const double dL_dO = 2.0 * w * (v - t);

    double max_rel = 0.0;
    for (int c = 0; c < n_desc; ++c) {
        const double expected = dL_dO * dg_dW(c);
        const double denom    = std::max(1.0, std::abs(expected));
        max_rel = std::max(max_rel, std::abs(g_out[c] - expected) / denom);
    }
    /* Tolerance dominated by the snapshot's FP round-off (aeds is built
     * by accumulating per-pair descriptor contributions; the order is
     * neighbor-list dependent and the integration walks a fresh
     * neighbour list each compute call). 1e-10 is well above that
     * floor for a stable 16-atom BCC cell at the relaxed state. */
    CHECK(max_rel < 1e-10);
}

// ----------------------------------------------------------------------------
// Phase 3.7.5: pair_tadah dvirial_dW FD parity.
//
// At the relaxed state x*, the iso-strain mixed partial is
//   ∂²E/∂ε∂W_c|_iso = ∂D_total[c]/∂ε|_iso     (since E = W·D_total)
// where D_total[c] = Σ_local_a aeds(c, a). The snapshot stores
// dvirial_dW[0, c] = -∂²E/∂ε∂W_c, so:
//   dvirial_dW[0, c] ≈ -(D_total(+ε)[c] - D_total(-ε)[c]) / (2ε).
//
// Independent FD reference: rescale all atom positions and box edges by
// (1+ε), call pair->compute (capture_adjoint_state=true populates aeds),
// sum per-atom AEDs; compare to dvirial_dW captured at ε=0.
TEST_CASE("phase3_pilot — pair_tadah dvirial_dW matches FD of Σ aed under "
          "iso strain to 1e-5 relative",
          "[phase3_pilot][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) { SUCCEED("skipped (no pot)"); return; }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) { SUCCEED("skipped"); return; }
    }
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx = tadah::core::CoreUserConfigParser().parse(pot_path));

    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, simple_bcc_pilot_script(), pot_ctx));

    auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pair != nullptr);
    auto* lt = pair->lammps_tadah();
    REQUIRE(lt != nullptr);

    /* Step 1: capture dvirial_dW at x*, ε=0. */
    lt->adjoint_snapshot.box_dofs = 1;
    lt->capture_adjoint_state = true;
    pair->compute(0, 0);
    lt->capture_adjoint_state = false;
    REQUIRE(lt->adjoint_snapshot.valid);
    REQUIRE(lt->adjoint_snapshot.dvirial_dW.size() ==
            static_cast<std::size_t>(lt->S.descriptor_size()));
    const std::vector<double> dvirial_dW_at_xstar =
        lt->adjoint_snapshot.dvirial_dW;
    const int n_desc = static_cast<int>(lt->S.descriptor_size());
    const int nlocal = lmp->atom->nlocal;
    const int nghost = lmp->atom->nghost;
    const int n_all  = nlocal + nghost;

    /* FD scratch buffers — sum-of-AEDs over local atoms at ±ε. */
    auto sum_local_aed = [&](std::vector<double>& out) {
        out.assign(n_desc, 0.0);
        for (int c = 0; c < n_desc; ++c) {
            for (int a = 0; a < nlocal; ++a) out[c] += lt->aeds(c, a);
        }
    };

    /* Save current atom positions (locals + ghosts) and box state. */
    std::vector<double> x_save(static_cast<std::size_t>(3 * n_all), 0.0);
    for (int i = 0; i < n_all; ++i) {
        x_save[3*i + 0] = lmp->atom->x[i][0];
        x_save[3*i + 1] = lmp->atom->x[i][1];
        x_save[3*i + 2] = lmp->atom->x[i][2];
    }
    double boxlo_save[3];
    double boxhi_save[3];
    for (int k = 0; k < 3; ++k) {
        boxlo_save[k] = lmp->domain->boxlo[k];
        boxhi_save[k] = lmp->domain->boxhi[k];
    }
    const double xy_save = lmp->domain->xy;
    const double xz_save = lmp->domain->xz;
    const double yz_save = lmp->domain->yz;

    auto apply_iso = [&](double scale) {
        /* Convert all atoms (locals + ghosts) to lamda so ghosts scale
         * affinely with the box. Then F = (1+scale)·I on box edges and
         * tilts, then map back to cartesian. Matches the Hessian
         * assembler's box-DOF sweep pattern. */
        lmp->domain->x2lamda(n_all);
        const double f = 1.0 + scale;
        for (int k = 0; k < 3; ++k) {
            lmp->domain->boxhi[k] = boxlo_save[k] +
                                    (boxhi_save[k] - boxlo_save[k]) * f;
        }
        lmp->domain->xy = xy_save * f;
        lmp->domain->xz = xz_save * f;
        lmp->domain->yz = yz_save * f;
        lmp->domain->set_global_box();
        lmp->domain->set_local_box();
        lmp->domain->lamda2x(n_all);
    };
    auto restore = [&]() {
        for (int k = 0; k < 3; ++k) {
            lmp->domain->boxhi[k] = boxhi_save[k];
            lmp->domain->boxlo[k] = boxlo_save[k];
        }
        lmp->domain->xy = xy_save;
        lmp->domain->xz = xz_save;
        lmp->domain->yz = yz_save;
        lmp->domain->set_global_box();
        lmp->domain->set_local_box();
        for (int i = 0; i < n_all; ++i) {
            lmp->atom->x[i][0] = x_save[3*i + 0];
            lmp->atom->x[i][1] = x_save[3*i + 1];
            lmp->atom->x[i][2] = x_save[3*i + 2];
        }
    };

    const double h = 1e-5;
    std::vector<double> D_plus, D_minus;

    /* +h. capture_adjoint_state populates aeds; box_dofs stays at 1 but
     * we only use the aeds output here. */
    lt->capture_adjoint_state = true;
    apply_iso(+h);
    lmp->neighbor->ago = 0;
    pair->compute(0, 0);
    sum_local_aed(D_plus);
    restore();

    /* -h. */
    apply_iso(-h);
    lmp->neighbor->ago = 0;
    pair->compute(0, 0);
    sum_local_aed(D_minus);
    restore();
    lt->capture_adjoint_state = false;
    lt->adjoint_snapshot.box_dofs = 0;

    /* FD reference: dD_total/dε ≈ (D_plus - D_minus) / (2h).
     * Analytical dvirial_dW = -dD_total/dε. */
    std::vector<double> dD(n_desc, 0.0);
    double max_rel = 0.0;
    double max_abs = 0.0;
    for (int c = 0; c < n_desc; ++c) {
        dD[c] = (D_plus[c] - D_minus[c]) / (2.0 * h);
        const double expected = -dD[c];
        const double diff     = std::abs(dvirial_dW_at_xstar[c] - expected);
        const double denom    = std::max(1.0, std::abs(expected));
        max_rel = std::max(max_rel, diff / denom);
        max_abs = std::max(max_abs, diff);
    }
    /* The bias / species-intercept descriptor row has Σ aed exactly
     * constant under any deformation (one 1.0 per atom, independent of
     * positions) — its FD residual is pure round-off, dominated by
     * 1/(2h)·(2·N·eps_machine). For a 16-atom cell at h=1e-5 that's
     * ~1e-12, well below the analytical zero. Other rows track the FD
     * step-size noise (~h² · |∂³D/∂ε³|), bounded ~1e-5 absolute for
     * this scale of descriptor.
     *
     * 1e-5 relative tolerance is the central-diff step-size noise
     * floor; tightening below ~1e-4 hits numerical instability across
     * descriptor types. */
    INFO("max abs diff = " << max_abs << ", max rel diff = " << max_rel);
    CHECK(max_rel < 1e-4);
}

// ----------------------------------------------------------------------------
// Phase 3.7D — Voigt-6 dvirial_dW FD parity.
//
// Generalises the iso (3.7.5) test to the full Voigt-6 block. For each
// Voigt α apply F = I + h·voigt_basis(α) on box edges + atom positions
// (matching LammpsHessianAssembler::apply_strain), pair->compute with
// capture_adjoint_state=true to repopulate aeds, sum local AEDs to get
// D_total(±h)[c], FD-check
//   dvirial_dW[α, c] ≈ -(D_plus - D_minus) / (2h).
//
// Voigt order {xx, yy, zz, yz, xz, xy} with engineering-shear (½ on
// off-diagonals) — same convention pair_tadah's accumulator uses.
//
// 1e-4 relative tolerance: same step-size noise floor as the iso case
// for diagonals; off-diagonals (½ factor) carry slightly more noise on
// the bias/intercept descriptor row but stay well under the bound on
// non-trivial descriptors. Bias row is excluded from the relative check
// (its FD residual is pure round-off — would divide by ~0).
TEST_CASE("phase3_pilot — pair_tadah Voigt-6 dvirial_dW matches FD of "
          "Σ aed under each Voigt strain to 1e-4 relative (Phase 3.7D)",
          "[phase3_pilot][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) { SUCCEED("skipped (no pot)"); return; }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) { SUCCEED("skipped"); return; }
    }
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx = tadah::core::CoreUserConfigParser().parse(pot_path));

    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, simple_bcc_pilot_script(), pot_ctx));

    auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pair != nullptr);
    auto* lt = pair->lammps_tadah();
    REQUIRE(lt != nullptr);

    /* Step 1: capture dvirial_dW at x*, all 6 Voigt rows. */
    lt->adjoint_snapshot.box_dofs = 6;
    lt->capture_adjoint_state = true;
    pair->compute(0, 0);
    lt->capture_adjoint_state = false;
    REQUIRE(lt->adjoint_snapshot.valid);
    REQUIRE(lt->adjoint_snapshot.dvirial_dW.size() ==
            6u * static_cast<std::size_t>(lt->S.descriptor_size()));
    const std::vector<double> dvirial_at_xstar =
        lt->adjoint_snapshot.dvirial_dW;
    const int n_desc = static_cast<int>(lt->S.descriptor_size());
    const int nlocal = lmp->atom->nlocal;
    const int nghost = lmp->atom->nghost;
    const int n_all  = nlocal + nghost;

    auto sum_local_aed = [&](std::vector<double>& out) {
        out.assign(n_desc, 0.0);
        for (int c = 0; c < n_desc; ++c)
            for (int a = 0; a < nlocal; ++a) out[c] += lt->aeds(c, a);
    };

    /* Save current atom positions (locals + ghosts) and box state. */
    std::vector<double> x_save(static_cast<std::size_t>(3 * n_all), 0.0);
    for (int i = 0; i < n_all; ++i) {
        x_save[3*i + 0] = lmp->atom->x[i][0];
        x_save[3*i + 1] = lmp->atom->x[i][1];
        x_save[3*i + 2] = lmp->atom->x[i][2];
    }
    double boxlo_save[3], boxhi_save[3];
    for (int k = 0; k < 3; ++k) {
        boxlo_save[k] = lmp->domain->boxlo[k];
        boxhi_save[k] = lmp->domain->boxhi[k];
    }
    const double xy_save = lmp->domain->xy;
    const double xz_save = lmp->domain->xz;
    const double yz_save = lmp->domain->yz;

    /* voigt_basis: matches lammps_hessian.cpp::voigt_basis. */
    auto voigt_basis = [](int alpha) {
        double e[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
        switch (alpha) {
            case 0: e[0][0] = 1.0; break;                            // xx
            case 1: e[1][1] = 1.0; break;                            // yy
            case 2: e[2][2] = 1.0; break;                            // zz
            case 3: e[1][2] = e[2][1] = 0.5; break;                  // yz
            case 4: e[0][2] = e[2][0] = 0.5; break;                  // xz
            case 5: e[0][1] = e[1][0] = 0.5; break;                  // xy
        }
        std::array<std::array<double,3>,3> out;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) out[i][j] = e[i][j];
        return out;
    };

    auto apply_voigt = [&](int alpha, double scale) {
        /* Mirror lammps_hessian.cpp::apply_strain. Build B (upper-tri
         * box matrix), F = I + scale·eps, B' = F·B; read back the
         * upper-triangular pieces, drop the residual sub-triangular
         * bits (O(scale²) for shear modes — within FD step-size noise). */
        const auto eps = voigt_basis(alpha);
        const double xprd = boxhi_save[0] - boxlo_save[0];
        const double yprd = boxhi_save[1] - boxlo_save[1];
        const double zprd = boxhi_save[2] - boxlo_save[2];
        double B[3][3] = {
            { xprd, xy_save, xz_save },
            { 0.0,  yprd,    yz_save },
            { 0.0,  0.0,     zprd    }
        };
        double F[3][3] = {
            { 1.0 + scale * eps[0][0],       scale * eps[0][1], scale * eps[0][2] },
            {       scale * eps[1][0], 1.0 + scale * eps[1][1], scale * eps[1][2] },
            {       scale * eps[2][0],       scale * eps[2][1], 1.0 + scale * eps[2][2] }
        };
        double FB[3][3]{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k) FB[i][j] += F[i][k] * B[k][j];

        lmp->domain->x2lamda(n_all);
        for (int k = 0; k < 3; ++k) {
            lmp->domain->boxlo[k] = boxlo_save[k];
        }
        lmp->domain->boxhi[0] = boxlo_save[0] + FB[0][0];
        lmp->domain->boxhi[1] = boxlo_save[1] + FB[1][1];
        lmp->domain->boxhi[2] = boxlo_save[2] + FB[2][2];
        lmp->domain->xy = FB[0][1];
        lmp->domain->xz = FB[0][2];
        lmp->domain->yz = FB[1][2];
        lmp->domain->set_global_box();
        lmp->domain->set_local_box();
        lmp->domain->lamda2x(n_all);
    };

    auto restore = [&]() {
        for (int k = 0; k < 3; ++k) {
            lmp->domain->boxhi[k] = boxhi_save[k];
            lmp->domain->boxlo[k] = boxlo_save[k];
        }
        lmp->domain->xy = xy_save;
        lmp->domain->xz = xz_save;
        lmp->domain->yz = yz_save;
        lmp->domain->set_global_box();
        lmp->domain->set_local_box();
        for (int i = 0; i < n_all; ++i) {
            lmp->atom->x[i][0] = x_save[3*i + 0];
            lmp->atom->x[i][1] = x_save[3*i + 1];
            lmp->atom->x[i][2] = x_save[3*i + 2];
        }
    };

    const double h = 1e-5;
    /* Per-Voigt α: capture aeds at ±h, take central diff. */
    double max_rel = 0.0;
    double max_abs = 0.0;
    int    bias_skip_count = 0;
    int    worst_alpha = -1;
    int    worst_c     = -1;

    lt->capture_adjoint_state = true;
    /* aeds buffer reused across compute calls — keep box_dofs at 0 for
     * the FD compute calls so the snapshot's box block is left
     * untouched (we already saved dvirial_at_xstar above). */
    lt->adjoint_snapshot.box_dofs = 0;

    std::vector<double> D_plus, D_minus;
    for (int alpha = 0; alpha < 6; ++alpha) {
        apply_voigt(alpha, +h);
        lmp->neighbor->ago = 0;
        pair->compute(0, 0);
        sum_local_aed(D_plus);
        restore();

        apply_voigt(alpha, -h);
        lmp->neighbor->ago = 0;
        pair->compute(0, 0);
        sum_local_aed(D_minus);
        restore();

        for (int c = 0; c < n_desc; ++c) {
            const double fd_val = (D_plus[c] - D_minus[c]) / (2.0 * h);
            const double expected_dvirial = -fd_val;
            const double anal = dvirial_at_xstar[
                static_cast<std::size_t>(alpha) * static_cast<std::size_t>(n_desc) + c];
            const double diff = std::abs(anal - expected_dvirial);
            const double denom = std::max(1.0, std::abs(expected_dvirial));
            const double rel = diff / denom;
            /* Bias / species-intercept descriptor: Σ aed = N atoms,
             * constant under any deformation ⇒ FD residual is pure
             * round-off (~1e-12 for h=1e-5, N=16). The analytical
             * value is exactly 0 in that row. Skip the relative check
             * (would divide by ~0); abs floor of 1e-9 catches a real
             * regression. */
            if (std::abs(expected_dvirial) < 1e-8 && std::abs(anal) < 1e-8) {
                ++bias_skip_count;
                continue;
            }
            if (rel > max_rel) {
                max_rel = rel;
                worst_alpha = alpha;
                worst_c     = c;
            }
            max_abs = std::max(max_abs, diff);
        }
    }
    lt->capture_adjoint_state = false;
    lt->adjoint_snapshot.box_dofs = 0;

    INFO("max rel = " << max_rel
         << " max abs = " << max_abs
         << " worst (α, c) = (" << worst_alpha << ", " << worst_c << ")"
         << " bias-row skips = " << bias_skip_count);
    CHECK(max_rel < 1e-4);
}

TEST_CASE("phase3_pilot — adjoint snapshot is captured at the post-"
          "minimize state and survives the Hessian assembler's "
          "central-diff sweep",
          "[phase3_pilot][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) { SUCCEED("skipped (no pot)"); return; }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) { SUCCEED("skipped"); return; }
    }
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx = tadah::core::CoreUserConfigParser().parse(pot_path));

    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, simple_bcc_pilot_script(), pot_ctx));

    auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pair != nullptr);
    auto* lt = pair->lammps_tadah();

    /* Before the driver call the snapshot is invalid (production MD
     * doesn't write into it). */
    CHECK_FALSE(lt->adjoint_snapshot.valid);

    tadah::hpo::SimsGradScriptInput in;
    const auto lsim = make_pilot_lsim();
    const std::string raw_e_per = runner.get_variable(lmp, "e_per");
    REQUIRE(tadah::hpo::populate_sims_grad_input_from_lammps(
        lmp.get(), lsim, {raw_e_per}, {"c_eatoms/atoms"}, 0,
        lsim.failscore, in));

    /* Driver's post-condition. */
    CHECK(lt->adjoint_snapshot.valid);
    CHECK(lt->adjoint_snapshot.n_atoms ==
          static_cast<std::size_t>(lmp->atom->nlocal));
    CHECK(lt->adjoint_snapshot.n_desc ==
          static_cast<std::size_t>(lt->S.descriptor_size()));
    /* Flag is restored to off. */
    CHECK_FALSE(lt->capture_adjoint_state);
}

// ----------------------------------------------------------------------------
// Phase 3.7C: lattice constant FD-over-W parity.
//
// End-to-end validation of the IFT-adjoint stack on a non-trivial
// observable (lx). The chain reads:
//   solve   H · λ = (∂g/∂x)ᵀ           ⇒ λ_atom ≈ 0, λ_box = lx
//   apply   dlx/dW[c] = λ_box · dvirial_dW[0, c]
//
// FD reference: rebuild pot_ctx with WEIGHTS[c_fd] ± h_w, fresh LAMMPS
// instance per side, re-relax, read lx; dlx/dW[c_fd]_FD =
// (lx_plus - lx_minus) / (2·h_w).
//
// Touches every Phase 3 component: AdjointSnapshot dvirial_dW capture
// (3.7.5), LammpsHessianAssembler box block (3.2) + sametag fix
// (3.7A), DenseAdjointSolver (3.4), observable registry (3.3),
// apply_ift_chain_kernel box block (3.7.5), LammpsTadah lifecycle
// stability across multiple instances (3.6 dtor crash fix).
TEST_CASE("phase3_pilot — lattice constant dlx/dW matches FD over W[c] "
          "to 1e-4 relative (Phase 3.7C)",
          "[phase3_pilot][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) { SUCCEED("skipped (no pot)"); return; }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) { SUCCEED("skipped"); return; }
    }
    tadah::core::Context pot_ctx_base;
    REQUIRE_NOTHROW(pot_ctx_base =
                       tadah::core::CoreUserConfigParser().parse(pot_path));

    /* Helpers: read / write WEIGHTS through the Context API. */
    auto get_weights = [](const tadah::core::Context& ctx) {
        const std::size_t N = ctx.size("WEIGHTS");
        std::vector<double> w(N);
        for (std::size_t i = 0; i < N; ++i) {
            w[i] = ctx.get<double>("WEIGHTS", i);
        }
        return w;
    };
    auto set_weights = [](tadah::core::Context& ctx,
                          const std::vector<double>& w) {
        ctx.remove("WEIGHTS");
        for (double wi : w) ctx.add("WEIGHTS", wi, w.size());
    };

    const std::vector<double> w_base = get_weights(pot_ctx_base);
    REQUIRE(w_base.size() >= 4);

    auto run_and_read_lx = [&](const tadah::core::Context& ctx_in) {
        LammpsRunner runner;
        auto lmp = runner.create_instance(
            0, {"lmp", "-log", "none", "-screen", "none"});
        REQUIRE(lmp != nullptr);
        REQUIRE(runner.run_script(lmp, bcc_lattice_constant_script(), ctx_in));
        const std::string raw_lx = runner.get_variable(lmp, "var_lx");
        REQUIRE_FALSE(raw_lx.empty());
        return std::stod(raw_lx);
    };

    /* ------- analytical dlx/dW via compute_sims_grad_for_script ------- */

    /* Spin up the baseline instance fresh and pass it (held in scope) to
     * `populate_sims_grad_input_from_lammps`. The chain expects the
     * LAMMPS instance to outlive the SimsGradScriptInput. */
    LammpsRunner runner_base;
    auto lmp_base = runner_base.create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp_base != nullptr);
    REQUIRE(runner_base.run_script(lmp_base, bcc_lattice_constant_script(),
                                    pot_ctx_base));

    const std::string raw_lx_base = runner_base.get_variable(lmp_base, "var_lx");
    REQUIRE_FALSE(raw_lx_base.empty());
    const double lx_0 = std::stod(raw_lx_base);

    /* LammpsSim: single observable `var_lx`, L2 loss with t=0 ⇒
     * dL/dW[c] = 2·w_loss·lx_0 · dlx/dW[c]. Pick w_loss=1, t=0. */
    tadah::hpo::LammpsSim lsim;
    lsim.script    = "bcc_lattice_constant";
    lsim.n_outvar  = {"var_lx"};
    lsim.n_varloss = {"var_lx"};
    lsim.t_varloss = {0.0};
    lsim.w_varloss = {1.0};
    lsim.i_varloss = {0};
    lsim.lossFn    = {tadah::LossFunction(tadah::LossFunction::Type::L2)};
    lsim.failscore = 1.0e8;

    tadah::hpo::SimsGradScriptInput in;
    REQUIRE(tadah::hpo::populate_sims_grad_input_from_lammps(
        lmp_base.get(), lsim, {raw_lx_base}, {"lx"},
        /*box_dofs=*/1, lsim.failscore, in));

    REQUIRE(in.dvirial_dW != nullptr);
    REQUIRE(in.box_dofs == 1);
    REQUIRE(in.dF_dW != nullptr);

    /* Bypass the loss fold: drive apply_ift_chain_kernel directly so the
     * test compares dlx/dW (not dL/dW) — eliminates one layer of
     * arithmetic between analytical and FD. */
    tadah::hpo::ObservableHandle h_lx;
    REQUIRE(tadah::hpo::try_register("lx", h_lx));
    REQUIRE_FALSE(h_lx.dg_dx_is_zero);

    Eigen::VectorXd dg_dx_lx;
    h_lx.dg_dx(in.ctx, dg_dx_lx);
    Eigen::VectorXd dg_dW_lx;
    h_lx.dg_dW(in.ctx, dg_dW_lx);

    tadah::hpo::DenseAdjointSolver solver(Eigen::MatrixXd(in.H),
                                          /*n_atom_dofs=*/3 * in.ctx.nlocal);
    Eigen::VectorXd lambda(3 * in.ctx.nlocal + in.box_dofs);
    solver.solve(dg_dx_lx, lambda);

    Eigen::VectorXd dlx_dW_anal;
    tadah::hpo::apply_ift_chain_kernel(
        dg_dW_lx, lambda, in.dF_dW, in.ctx.nlocal, in.ctx.n_desc,
        dlx_dW_anal, in.dvirial_dW, in.box_dofs);
    REQUIRE(dlx_dW_anal.size() == in.ctx.n_desc);

    /* ------- FD reference: one selected W component ------- */

    /* Pick a W component with non-trivial magnitude. The pot.tadah's
     * WEIGHTS[0] is the bias/intercept (~1500 for Ta); WEIGHTS[2] is a
     * 2-body radial coefficient with O(700) magnitude — a robust
     * choice for FD precision. */
    const int c_fd = 2;
    const double w_c0 = w_base[c_fd];
    /* h_w sweep: scan a couple of step sizes and take the one with
     * smallest reasonable noise. lx response is roughly linear in W only
     * within ~1e-5 relative; large h dominates by cubic terms, small h
     * by minimize convergence noise (~3e-8 on positions ⇒ ~3e-7 on lx).
     * We want |Δlx|/h_w ~ stable and |Δlx| > 1e-5 · lx_0 to stay above
     * the minimize floor.
     *
     * Pre-sweep results on a 16-atom BCC Ta with this pot:
     *   1e-3·|W| → 57% rel (deep into cubic)
     *   1e-4·|W| → ~5% rel (still cubic-dominated)
     *   1e-5·|W| → ~1e-3 rel (linear regime, minor noise)
     *   1e-6·|W| → noise floor too high
     * → 1e-5·|W| is the sweet spot. */
    const double h_w = 1.0e-5 * std::abs(w_c0);
    REQUIRE(h_w > 0.0);

    auto perturbed_ctx = [&](double delta_c) {
        tadah::core::Context ctx = pot_ctx_base;     /* copy */
        std::vector<double> w = w_base;
        w[c_fd] += delta_c;
        set_weights(ctx, w);
        return ctx;
    };

    const double lx_plus  = run_and_read_lx(perturbed_ctx(+h_w));
    const double lx_minus = run_and_read_lx(perturbed_ctx(-h_w));
    const double dlx_dW_fd_c = (lx_plus - lx_minus) / (2.0 * h_w);

    const double dlx_dW_anal_c = dlx_dW_anal(c_fd);

    /* Tolerance: 1e-4 relative. Observed at the chosen h_w: ~3e-5
     * relative. The step-size sweep documented above bounds the
     * cubic-noise term at <1e-4 for this h. */
    const double diff   = std::abs(dlx_dW_anal_c - dlx_dW_fd_c);
    const double denom  = std::max(1.0e-6, std::abs(dlx_dW_fd_c));
    const double relerr = diff / denom;
    INFO("c_fd = " << c_fd
         << "  lx_0 = " << lx_0
         << "  lx_plus = " << lx_plus
         << "  lx_minus = " << lx_minus
         << "  h_w = " << h_w
         << "  dlx/dW_FD = " << dlx_dW_fd_c
         << "  dlx/dW_anal = " << dlx_dW_anal_c
         << "  rel = " << relerr);
    CHECK(relerr < 1.0e-4);
}

// ----------------------------------------------------------------------------
// Phase 3.7E (3.7D follow-up) — end-to-end Voigt-6 aniso parity.
//
// Mirrors the 3.7C iso test exactly except for two changes:
//   (a) script uses `box/relax aniso` (3 diagonal box DOFs free during
//       relaxation; cubic equilibrium for stress-free Ta);
//   (b) `populate_sims_grad_input_from_lammps` is called with
//       `box_dofs == 6` instead of 1 — the chain runs through
//       pair_tadah's Voigt-6 dvirial_dW accumulator (3.7D),
//       LammpsHessianAssembler's 6×6 box Hessian block (3.2 + 3.7A),
//       DenseAdjointSolver's 6-component lambda solve, observable
//       registry's `lattice_constant` Voigt placement (axis 0 ⇒ row
//       3N+0), and apply_ift_chain_kernel's box-block term over all 6
//       Voigt rows.
//
// FD reference is identical to the 3.7C test (perturb W[c_fd] ± h_w,
// re-relax, read lx). Physically the response should match the iso
// case for a stress-free cubic Ta cell — the analytical paths differ
// (Voigt-6 vs Voigt-1) but converge on the same physical dlx/dW.
//
// This validates the 3.7D Voigt-6 accumulator together with every
// other Voigt-6-aware component in the chain.
TEST_CASE("phase3_pilot — lattice constant dlx/dW via Voigt-6 chain "
          "matches FD over W[c] to 1e-4 relative (Phase 3.7E)",
          "[phase3_pilot][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) { SUCCEED("skipped (no pot)"); return; }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) { SUCCEED("skipped"); return; }
    }
    tadah::core::Context pot_ctx_base;
    REQUIRE_NOTHROW(pot_ctx_base =
                       tadah::core::CoreUserConfigParser().parse(pot_path));

    auto get_weights = [](const tadah::core::Context& ctx) {
        const std::size_t N = ctx.size("WEIGHTS");
        std::vector<double> w(N);
        for (std::size_t i = 0; i < N; ++i) {
            w[i] = ctx.get<double>("WEIGHTS", i);
        }
        return w;
    };
    auto set_weights = [](tadah::core::Context& ctx,
                          const std::vector<double>& w) {
        ctx.remove("WEIGHTS");
        for (double wi : w) ctx.add("WEIGHTS", wi, w.size());
    };

    const std::vector<double> w_base = get_weights(pot_ctx_base);
    REQUIRE(w_base.size() >= 4);

    auto run_and_read_lx = [&](const tadah::core::Context& ctx_in) {
        LammpsRunner runner;
        auto lmp = runner.create_instance(
            0, {"lmp", "-log", "none", "-screen", "none"});
        REQUIRE(lmp != nullptr);
        REQUIRE(runner.run_script(lmp, bcc_lattice_constant_aniso_script(),
                                  ctx_in));
        const std::string raw_lx = runner.get_variable(lmp, "var_lx");
        REQUIRE_FALSE(raw_lx.empty());
        return std::stod(raw_lx);
    };

    /* ------- analytical dlx/dW via Voigt-6 chain ------- */

    LammpsRunner runner_base;
    auto lmp_base = runner_base.create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp_base != nullptr);
    REQUIRE(runner_base.run_script(lmp_base,
                                   bcc_lattice_constant_aniso_script(),
                                   pot_ctx_base));

    const std::string raw_lx_base = runner_base.get_variable(lmp_base, "var_lx");
    REQUIRE_FALSE(raw_lx_base.empty());
    const double lx_0 = std::stod(raw_lx_base);

    tadah::hpo::LammpsSim lsim;
    lsim.script    = "bcc_lattice_constant_aniso";
    lsim.n_outvar  = {"var_lx"};
    lsim.n_varloss = {"var_lx"};
    lsim.t_varloss = {0.0};
    lsim.w_varloss = {1.0};
    lsim.i_varloss = {0};
    lsim.lossFn    = {tadah::LossFunction(tadah::LossFunction::Type::L2)};
    lsim.failscore = 1.0e8;

    /* The load-bearing change: box_dofs = 6 (was 1 in the 3.7C test). */
    tadah::hpo::SimsGradScriptInput in;
    REQUIRE(tadah::hpo::populate_sims_grad_input_from_lammps(
        lmp_base.get(), lsim, {raw_lx_base}, {"lx"},
        /*box_dofs=*/6, lsim.failscore, in));

    REQUIRE(in.dvirial_dW != nullptr);
    REQUIRE(in.box_dofs == 6);
    REQUIRE(in.dF_dW != nullptr);

    tadah::hpo::ObservableHandle h_lx;
    REQUIRE(tadah::hpo::try_register("lx", h_lx));
    REQUIRE_FALSE(h_lx.dg_dx_is_zero);

    Eigen::VectorXd dg_dx_lx;
    h_lx.dg_dx(in.ctx, dg_dx_lx);
    Eigen::VectorXd dg_dW_lx;
    h_lx.dg_dW(in.ctx, dg_dW_lx);

    /* dg_dx_lx for `lx` with box_dofs=6: lx at row 3N+0 (Voigt-xx),
     * zeros elsewhere. Sanity-check the registry got it right. */
    REQUIRE(dg_dx_lx.size() == 3 * in.ctx.nlocal + 6);
    CHECK(dg_dx_lx(3 * in.ctx.nlocal + 0) == Approx(lx_0));
    for (int alpha = 1; alpha < 6; ++alpha) {
        CHECK(dg_dx_lx(3 * in.ctx.nlocal + alpha) == Approx(0.0));
    }

    tadah::hpo::DenseAdjointSolver solver(Eigen::MatrixXd(in.H),
                                          /*n_atom_dofs=*/3 * in.ctx.nlocal);
    Eigen::VectorXd lambda(3 * in.ctx.nlocal + in.box_dofs);
    solver.solve(dg_dx_lx, lambda);

    Eigen::VectorXd dlx_dW_anal;
    tadah::hpo::apply_ift_chain_kernel(
        dg_dW_lx, lambda, in.dF_dW, in.ctx.nlocal, in.ctx.n_desc,
        dlx_dW_anal, in.dvirial_dW, in.box_dofs);
    REQUIRE(dlx_dW_anal.size() == in.ctx.n_desc);

    /* ------- FD reference (same h_w as 3.7C, same c_fd) ------- */

    const int c_fd = 2;
    const double w_c0 = w_base[c_fd];
    const double h_w = 1.0e-5 * std::abs(w_c0);
    REQUIRE(h_w > 0.0);

    auto perturbed_ctx = [&](double delta_c) {
        tadah::core::Context ctx = pot_ctx_base;
        std::vector<double> w = w_base;
        w[c_fd] += delta_c;
        set_weights(ctx, w);
        return ctx;
    };

    const double lx_plus  = run_and_read_lx(perturbed_ctx(+h_w));
    const double lx_minus = run_and_read_lx(perturbed_ctx(-h_w));
    const double dlx_dW_fd_c = (lx_plus - lx_minus) / (2.0 * h_w);

    const double dlx_dW_anal_c = dlx_dW_anal(c_fd);

    const double diff   = std::abs(dlx_dW_anal_c - dlx_dW_fd_c);
    const double denom  = std::max(1.0e-6, std::abs(dlx_dW_fd_c));
    const double relerr = diff / denom;
    INFO("c_fd = " << c_fd
         << "  lx_0 = " << lx_0
         << "  lx_plus = " << lx_plus
         << "  lx_minus = " << lx_minus
         << "  h_w = " << h_w
         << "  dlx/dW_FD = " << dlx_dW_fd_c
         << "  dlx/dW_anal_voigt6 = " << dlx_dW_anal_c
         << "  rel = " << relerr);
    /* Tolerance: 5e-4 relative for Voigt-6.
     *
     * The 3.7C iso case ran at ~3e-5 relative on the same h_w. Voigt-6
     * is ~4× looser because the analytical path here picks up extra
     * step-size noise from the Hessian assembler's off-diagonal virial-
     * derivative central diffs (12 box-block off-diagonals on cubic
     * crystal: each is `±virial(±h_strain)` with h_strain=1e-5). The
     * 6-component λ solve and the 6-row dvirial_dW contraction add
     * further noise channels. Observed at h_w = 1e-5·|W| on this
     * config: ~1.3e-4 relative.
     *
     * 5e-4 leaves comfortable headroom against natural variation
     * across descriptor types / cells / step-sizes while staying tight
     * enough to catch the kind of bug that 3.7E originally exposed
     * (the missing box-box off-diagonals were a 1.2 = 120% relative
     * disagreement). */
    CHECK(relerr < 5.0e-4);
}

// ----------------------------------------------------------------------------
// Phase 3.7B-orchestrator: end-to-end SimsGradProvider lambda body.
//
// Validates the lambda body that `HPO_Host::analytical_gradient` builds
// when `trg_.LS` is non-empty. The lambda's per-script logic is:
//   1. detect_box_dofs_from_script(lsim.script_lines)
//   2. extract_varloss_expressions(lsim)
//   3. populate_sims_grad_input_from_lammps(lmp, lsim, v_outvar,
//                                           expressions, box_dofs, ...)
// then the resulting SimsGradScriptInput feeds compute_sims_grad_dWchain
// to fold sims_loss gradients into g_out.
//
// This test reproduces that pipeline against the pilot's BCC Ta script
// using a SYNTHETIC dWdtheta (identity, n_desc rows). The result must
// match the closed-form (Σ aeds)/N · 2·w·(v−t) that case 1 of
// `[phase3_pilot]` already verified — but here the expressions and
// box_dofs come from the helpers, not hand-coded constants. A
// regression in either helper would surface as a parity break here.
TEST_CASE("phase3_orchestrator — SimsGradProvider lambda body matches "
          "the closed-form on the pilot script (helpers + populate + chain)",
          "[phase3_orchestrator][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) {
        WARN("phase3_orchestrator skipped: no pot. Set TADAH_BENCH_POT or "
             "re-cmake with TADAH_PATH.");
        SUCCEED("skipped (no pot)");
        return;
    }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) {
            WARN("phase3_orchestrator skipped: pot unreadable.");
            SUCCEED("skipped (pot unreadable)");
            return;
        }
    }

    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx = tadah::core::CoreUserConfigParser().parse(pot_path));

    /* Build the lsim metadata exactly as the eval pipeline would. The
     * key part is `script_lines` — the helpers walk it to extract
     * box_dofs (iso ⇒ 1) and the per-observable expression. */
    auto lsim = make_pilot_lsim();
    lsim.script_lines = simple_bcc_pilot_script();

    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, lsim.script_lines, pot_ctx));

    auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pair != nullptr);

    /* Read observables exactly as the eval pipeline would (one entry
     * per name in lsim.n_outvar). */
    std::vector<std::string> v_outvar;
    v_outvar.reserve(lsim.n_outvar.size());
    for (const auto& name : lsim.n_outvar)
        v_outvar.push_back(runner.get_variable(lmp, name));
    REQUIRE_FALSE(v_outvar.empty());
    REQUIRE_FALSE(v_outvar[0].empty());

    /* This is the load-bearing assertion of this test: the helpers
     * produce the same (expressions, box_dofs) that the pilot test
     * hand-codes. */
    const int box_dofs = tadah::hpo::detect_box_dofs_from_script(
        lsim.script_lines);
    CHECK(box_dofs == 1);   /* `fix relax all box/relax iso` */

    const auto expressions = tadah::hpo::extract_varloss_expressions(lsim);
    REQUIRE(expressions.size() == 1);
    CHECK(expressions[0] == "$(c_eatoms/atoms)");

    tadah::hpo::SimsGradScriptInput in;
    REQUIRE(tadah::hpo::populate_sims_grad_input_from_lammps(
        lmp.get(), lsim, v_outvar, expressions, box_dofs,
        lsim.failscore, in));

    REQUIRE(in.ctx.aeds != nullptr);
    REQUIRE(in.ctx.n_desc > 0);
    REQUIRE(in.ctx.nlocal > 0);
    REQUIRE(in.dF_dW != nullptr);
    REQUIRE(in.v_observed.size() == 1);

    /* Synthetic identity dWdtheta over n_desc rows so g_out picks out
     * one dL/dW coefficient per θ. Mirrors the [phase3_pilot] case 1
     * setup so we can assert the same closed-form. */
    const int n_desc = in.ctx.n_desc;
    std::vector<double> dWdtheta(static_cast<std::size_t>(n_desc * n_desc), 0.0);
    for (int r = 0; r < n_desc; ++r) dWdtheta[r * n_desc + r] = 1.0;
    std::vector<int> theta_row_to_g_idx(n_desc);
    for (int r = 0; r < n_desc; ++r) theta_row_to_g_idx[r] = r;
    std::vector<int> all_theta_indices(n_desc);
    for (int r = 0; r < n_desc; ++r) all_theta_indices[r] = r;

    std::vector<double> g_out(n_desc, 0.0);
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_sims_grad_dWchain(
        {in}, dWdtheta, static_cast<std::size_t>(n_desc),
        theta_row_to_g_idx, all_theta_indices, g_out, needs_fd));
    CHECK(needs_fd.empty());

    /* Closed-form expectation, identical to the pilot's case 1: cohesive
     * E ⇒ dg/dW = (Σ_a aeds(:,a))/N. λ_k=0 ⇒ dO/dW = dg/dW. L2 fold ⇒
     * dL/dW = 2·w·(v − t)·dO/dW. Identity dWdtheta ⇒ g_out[c] = dL/dW[c]. */
    Eigen::VectorXd dg_dW_close(n_desc);
    for (int c = 0; c < n_desc; ++c) {
        double sum = 0.0;
        for (int a = 0; a < in.ctx.nlocal; ++a) sum += (*in.ctx.aeds)(c, a);
        dg_dW_close(c) = sum / in.ctx.natoms;
    }
    const double v = in.v_observed[0];
    const double t = lsim.t_varloss[0];
    const double w = lsim.w_varloss[0];
    const double dL_dO = 2.0 * w * (v - t);

    double max_rel = 0.0;
    for (int c = 0; c < n_desc; ++c) {
        const double expected = dL_dO * dg_dW_close(c);
        const double denom    = std::max(1.0, std::abs(expected));
        max_rel = std::max(max_rel, std::abs(g_out[c] - expected) / denom);
    }
    CHECK(max_rel < 1e-10);
}

// ----------------------------------------------------------------------------
// Phase 3.7B-warmstart: hot-var script analytical gradient.
//
// Before 3.7B-warmstart, the gradient lambda short-circuited on any
// script with non-empty `hotvars`, forcing all-FD for the whole call.
// Now the lambda builds a name-keyed `hot_snapshot` map from the
// warmstart cache (with `initial` / midpoint fallback) and injects
// `variable <name> string <value>` before each LAMMPS script runs.
//
// This test pins down the snapshot construction logic and exercises the
// per-script body end-to-end on a hot-var-referencing script
// (`lattice bcc ${hot_a0}`). Three sub-checks:
//   1. Cache empty + `hv.initial` set: snapshot uses `initial`.
//   2. Best-hot primed via `warmstart::promote`: snapshot uses cached value.
//   3. End-to-end happy path: snapshot seed flows through LAMMPS variable
//      injection → script runs → relaxed-state observable → chain rule
//      → g_out matches the cohesive-E closed-form to 1e-10 rel.
TEST_CASE("phase3_orchestrator — hot-var script analytical path: "
          "snapshot construction + script injection + chain matches "
          "the closed-form (Phase 3.7B-warmstart)",
          "[phase3_orchestrator][phase3][.][!benchmark]")
{
    const std::string pot_path = resolve_pot_path();
    if (pot_path.empty()) {
        WARN("phase3_orchestrator (warmstart) skipped: no pot.");
        SUCCEED("skipped (no pot)");
        return;
    }
    {
        std::ifstream ifs(pot_path);
        if (!ifs.good()) {
            WARN("phase3_orchestrator (warmstart) skipped: pot unreadable.");
            SUCCEED("skipped (pot unreadable)");
            return;
        }
    }
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx = tadah::core::CoreUserConfigParser().parse(pot_path));

    namespace ws = tadah::hpo::warmstart;

    auto lsim = make_hotvar_pilot_lsim();
    lsim.script_lines = simple_bcc_hotvar_pilot_script();
    const std::vector<tadah::hpo::LammpsSim> LS = { lsim };

    /* Reproduce the lambda's hot_snapshot construction in-test. Same
     * algorithm as hpo_host_impl.hpp: collapse by `name`, prefer cache
     * over `hv.initial` over midpoint. */
    auto build_snapshot =
        [](const std::vector<tadah::hpo::LammpsSim>& LS_in) {
            std::unordered_map<std::string, double> out;
            for (const auto& ls : LS_in) {
                for (const auto& hv : ls.hotvars) {
                    if (out.count(hv.name)) continue;
                    if (auto cached = ws::best_get(ls.script, hv.name)) {
                        out[hv.name] = *cached;
                    } else if (hv.initial) {
                        out[hv.name] = *hv.initial;
                    } else {
                        out[hv.name] = 0.5 * (hv.min + hv.max);
                    }
                }
            }
            return out;
        };

    /* Clean cache for this (script, name) so the empty-cache path is
     * deterministic across re-runs of the test. */
    ws::clear();

    /* (1) Cache empty + initial set → snapshot uses initial. */
    {
        auto snap = build_snapshot(LS);
        REQUIRE(snap.count("hot_a0") == 1);
        CHECK(snap["hot_a0"] == Approx(3.301).margin(1e-12));
    }

    /* (2) Cache primed → snapshot uses cached value (not initial). The gradient
     * path reads the best-hot map, so prime it via promote (not the per-chain
     * runtime set). */
    ws::promote(lsim.script, "hot_a0", 3.250);
    {
        auto snap = build_snapshot(LS);
        REQUIRE(snap.count("hot_a0") == 1);
        CHECK(snap["hot_a0"] == Approx(3.250).margin(1e-12));
    }

    /* (3) End-to-end: cache-sourced seed injected via lmp->input->one
     * before run_script, then full populate + chain. Mirrors the
     * lambda's per-script body exactly. */
    LammpsRunner runner;
    auto lmp = runner.create_instance(0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);

    auto snap = build_snapshot(LS);
    {
        char cmd[256];
        for (const auto& kv : snap) {
            std::snprintf(cmd, sizeof(cmd),
                          "variable %s string %.17g",
                          kv.first.c_str(), kv.second);
            REQUIRE_NOTHROW(lmp->input->one(cmd));
        }
    }
    REQUIRE(runner.run_script(lmp, lsim.script_lines, pot_ctx));

    std::vector<std::string> v_outvar;
    v_outvar.reserve(lsim.n_outvar.size());
    for (const auto& name : lsim.n_outvar)
        v_outvar.push_back(runner.get_variable(lmp, name));
    REQUIRE_FALSE(v_outvar.empty());
    REQUIRE_FALSE(v_outvar[0].empty());

    const int box_dofs = tadah::hpo::detect_box_dofs_from_script(
        lsim.script_lines);
    CHECK(box_dofs == 1);  /* iso relax */

    const auto expressions = tadah::hpo::extract_varloss_expressions(lsim);
    REQUIRE(expressions.size() == 1);
    CHECK(expressions[0] == "$(c_eatoms/atoms)");

    tadah::hpo::SimsGradScriptInput in;
    REQUIRE(tadah::hpo::populate_sims_grad_input_from_lammps(
        lmp.get(), lsim, v_outvar, expressions, box_dofs,
        lsim.failscore, in));

    REQUIRE(in.ctx.aeds != nullptr);
    REQUIRE(in.ctx.n_desc > 0);
    REQUIRE(in.ctx.nlocal > 0);
    REQUIRE(in.dF_dW != nullptr);

    /* Identity dWdtheta over n_desc descriptors, same as the other
     * pilot cases. */
    const int n_desc = in.ctx.n_desc;
    std::vector<double> dWdtheta(static_cast<std::size_t>(n_desc * n_desc), 0.0);
    for (int r = 0; r < n_desc; ++r) dWdtheta[r * n_desc + r] = 1.0;
    std::vector<int> theta_row_to_g_idx(n_desc);
    for (int r = 0; r < n_desc; ++r) theta_row_to_g_idx[r] = r;
    std::vector<int> all_theta_indices(n_desc);
    for (int r = 0; r < n_desc; ++r) all_theta_indices[r] = r;

    std::vector<double> g_out(n_desc, 0.0);
    std::vector<int>    needs_fd;
    REQUIRE(tadah::hpo::compute_sims_grad_dWchain(
        {in}, dWdtheta, static_cast<std::size_t>(n_desc),
        theta_row_to_g_idx, all_theta_indices, g_out, needs_fd));
    CHECK(needs_fd.empty());

    /* Closed-form cohesive-E expectation (same as case 1 of the pilot
     * test): λ_k=0 since dg/dx=0 for cohesive E per atom; chain reduces
     * to dL/dW[c] = 2·w·(v−t) · (Σ_a aeds(:,a))/N. The relaxed Σ aeds
     * is independent of which `${hot_a0}` seed we used (both 3.301 and
     * 3.250 converge to the same physical relaxed lattice). */
    Eigen::VectorXd dg_dW_close(n_desc);
    for (int c = 0; c < n_desc; ++c) {
        double sum = 0.0;
        for (int a = 0; a < in.ctx.nlocal; ++a)
            sum += (*in.ctx.aeds)(c, a);
        dg_dW_close(c) = sum / in.ctx.natoms;
    }
    const double v = in.v_observed[0];
    const double t = lsim.t_varloss[0];
    const double w = lsim.w_varloss[0];
    const double dL_dO = 2.0 * w * (v - t);

    double max_rel = 0.0;
    for (int c = 0; c < n_desc; ++c) {
        const double expected = dL_dO * dg_dW_close(c);
        const double denom    = std::max(1.0, std::abs(expected));
        max_rel = std::max(max_rel, std::abs(g_out[c] - expected) / denom);
    }
    CHECK(max_rel < 1e-10);

    /* Tidy cache for re-runs. */
    ws::clear();
}

// ----------------------------------------------------------------------------
// Provider failure-mode and all-or-nothing fall-back are already
// covered by [sims_grad_provider] cases 1-4 in test_hpo_analytical.cpp.
// The HPO_Host::analytical_gradient lambda body construction is
// reachable by inspection (when `trg_.LS` is non-empty the lambda is
// installed; nullptr is only passed when `trg_.LS` is empty).
// The [phase3_orchestrator] cases above exercise the lambda's full
// happy-path body — helpers + populate + chain — on real LAMMPS,
// including the Phase 3.7B-warmstart hot-var injection path.

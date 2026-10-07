// Phase 3.2 — LammpsHessianAssembler tests.
//
// These tests instantiate a real LAMMPS+pair_tadah on a 4-atom BCC Ta
// cell, minimise to the equilibrium configuration, then assemble H via
// central diff and assert structural invariants (symmetry, acoustic
// zero modes, positive interior spectrum) and a self-consistency check
// (eigenvector probe matches its eigenvalue under independent FD).
//
// The potential (bench_pot.h): TADAH_BENCH_POT when set, else the one
// shipped with the tests, tests/data/bench_ta.tadah (the Quick Start Ta
// potential; its header gives its provenance). One that cannot be read,
// parsed or run is a failure; there is no skip.
//
// Tags: every LAMMPS-using case carries `[.][!benchmark]` (hidden, so not
// discovered) and runs in the CTest entry hpo_lammps_hessian (label bench),
// `test_hpo "[lammps_hessian][.]"`, one process for the seven cases. They
// were hidden because of a crash when LAMMPS with pair_style tadah was
// destroyed; that crash is gone (the seven cases, and the eight of
// test_hpo_phase3_pilot.cpp, create and destroy their LAMMPS instances in one
// process). The two LAMMPS-free cases at the end are discovered as usual.
//
// To run the hidden cases by hand:
//   build_tadah/_deps/tadah.hpo-build/tests/test_hpo "[lammps_hessian][.]"

#include "catch2/catch.hpp"

#include "bench_pot.h"

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/lammps_hessian.h>
#include <tadah/hpo/runners.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <atom.h>
#include <comm.h>
#include <domain.h>
#include <force.h>
#include <neighbor.h>
#include <pair.h>
#include <pair_tadah.h>

#include <cstring>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 4-atom bcc Ta: 1×1×2 supercell (BCC unit has 2 atoms ⇒ 4 atoms total).
// a0 is intentionally a few percent off the relaxed value so that the
// `minimize` call following pair_coeff exercises the static-relax path
// the IFT-adjoint targets.
std::vector<std::string> bcc_ta_4atom_script() {
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 3.301",
        "region box block 0 1 0 1 0 2 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "fix relax all box/relax iso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 1e-12 1e-12 500 500",
        "unfix relax",
        "run 0",
    };
}

/* LAMMPS with pair_style tadah on the relaxed 4-atom bcc Ta cell. A
 * potential that does not parse, a LAMMPS instance that cannot be created
 * or a script that fails is a failure: the constructor throws, and Catch
 * reports the exception. */
struct LammpsFixture {
    LammpsRunner            runner;
    LammpsRunner::LMPPtr    lmp;
    LAMMPS_NS::PairTadah*   pair = nullptr;

    explicit LammpsFixture(const std::string& pot_path) {
        tadah::core::PeriodicTable::initialize();   // ATOM symbols of the pot
        const tadah::core::Context pot_ctx =
            tadah::core::CoreUserConfigParser().parse(pot_path);

        lmp = runner.create_instance(
            0, {"lmp", "-log", "none", "-screen", "none"});
        if (!lmp)
            throw std::runtime_error(
                "LammpsFixture: cannot create a LAMMPS instance");
        if (!runner.run_script(lmp, bcc_ta_4atom_script(), pot_ctx))
            throw std::runtime_error(
                "LammpsFixture: the 4-atom bcc Ta script failed with the "
                "potential " + pot_path);
        pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
        if (!pair)
            throw std::runtime_error(
                "LammpsFixture: the pair style is not pair_style tadah");
    }
};

}  // namespace

TEST_CASE("LammpsHessianAssembler — atom-only block is symmetric and "
          "has 3 near-zero acoustic modes (4-atom BCC Ta)",
          "[lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());

    const int nlocal = fx.lmp->atom->nlocal;
    REQUIRE(nlocal == 4);

    tadah::hpo::LammpsHessianAssembler asm_;
    asm_.set_tikhonov(0.0);  // un-regularised for the symmetry/eigen check
    Eigen::MatrixXd H = asm_.assemble(fx.lmp.get(), /*box_dofs=*/0);

    REQUIRE(H.rows() == 3 * nlocal);
    REQUIRE(H.cols() == 3 * nlocal);

    // Symmetry tolerance: a 4-atom BCC Hessian has entries on the order
    // of 1e2 eV/Å²; the central-diff step h=1e-4 gives truncation noise
    // ~1e-4 in absolute terms; the post-call symmetrise(0.5*(H+Hᵀ))
    // collapses asymmetry. Tolerance 1e-9 catches transcription bugs
    // without flagging legitimate FP drift.
    const double asym = (H - H.transpose()).cwiseAbs().maxCoeff();
    CHECK(asym < 1e-9);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    REQUIRE(es.info() == Eigen::Success);
    Eigen::VectorXd evals = es.eigenvalues();
    std::sort(evals.data(), evals.data() + evals.size());

    // Acoustic translation modes: bulk Ta cell has 3 of them at exactly
    // zero, up to central-diff noise (~h² × stiffness ≈ 1e-6).
    for (int i = 0; i < 3; ++i) {
        CHECK(std::abs(evals(i)) < 1e-4);
    }
    // Remaining modes are optical/short-wavelength phonons; all should
    // be ≥ 0 at the relaxed configuration. Allow a wider negative
    // tolerance to swallow numerical noise on near-zero modes.
    for (int i = 3; i < evals.size(); ++i) {
        CHECK(evals(i) > -1e-4);
    }
    // At least one real positive mode must exist.
    CHECK(evals(evals.size() - 1) > 1.0);
}

TEST_CASE("LammpsHessianAssembler — Hessian-vector probe matches force "
          "central diff along a non-trivial direction",
          "[lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());
    const int nlocal = fx.lmp->atom->nlocal;
    REQUIRE(nlocal == 4);

    tadah::hpo::LammpsHessianAssembler asm_;
    asm_.set_tikhonov(0.0);
    Eigen::MatrixXd H = asm_.assemble(fx.lmp.get(), 0);

    // Construct a unit displacement direction v that's not aligned with
    // any acoustic mode (acoustic = constant-vector, so mix the
    // per-atom contributions).
    Eigen::VectorXd v = Eigen::VectorXd::Zero(3 * nlocal);
    v(0) = 1.0; v(4) = -0.7; v(8) = 0.3; v(11) = -0.6;
    v.normalize();

    // External FD: F(x* + ε·v) - F(x* - ε·v) over 2ε ⇒ -H·v.
    //
    // PBC ghost handling: like the assembler itself, we must shift each
    // local atom AND its periodic ghost images (atom->sametag chain) in
    // lockstep — otherwise pair_tadah's neighbor loop reads stale ghost
    // positions and the FD computes a different (non-physical) H than
    // the one the assembler produces. Mirrors PHONON's displace_atom.
    const double eps = 2.0e-4;
    const auto& runner = fx.runner;
    (void)runner;
    LAMMPS_NS::Atom* atom = fx.lmp->atom;
    const int nghost = atom->nghost;
    const int n_all  = nlocal + nghost;
    int* sametag     = atom->sametag;

    std::vector<double> x_save(3 * n_all);
    for (int i = 0; i < n_all; ++i) {
        x_save[3*i + 0] = atom->x[i][0];
        x_save[3*i + 1] = atom->x[i][1];
        x_save[3*i + 2] = atom->x[i][2];
    }

    auto displace_and_force = [&](double scale, std::vector<double>& out) {
        /* Reset to x* (locals + ghosts) before applying the new shift. */
        for (int i = 0; i < n_all; ++i) {
            atom->x[i][0] = x_save[3*i + 0];
            atom->x[i][1] = x_save[3*i + 1];
            atom->x[i][2] = x_save[3*i + 2];
        }
        for (int i = 0; i < nlocal; ++i) {
            for (int k = 0; k < 3; ++k) {
                const double delta = scale * v(3*i + k);
                atom->x[i][k] += delta;
                if (!sametag) continue;
                int j = i;
                while (sametag[j] >= 0) {
                    j = sametag[j];
                    atom->x[j][k] += delta;
                }
            }
        }
        /* pair->compute does NOT clear atom->f (it accumulates via +=);
         * the FD comparison reads the forces produced by ONE compute, so
         * the buffer must be zeroed first or successive calls poison each
         * other. ev_init won't help — vflag=0 leaves the force array
         * alone. Mirror the assembler's compute_clear_and_eval idiom. */
        fx.lmp->neighbor->ago = 0;
        const int nlocal_f = atom->nlocal;
        std::size_t nbytes = sizeof(double) * nlocal_f;
        if (fx.lmp->force->newton) nbytes += sizeof(double) * atom->nghost;
        if (nbytes) std::memset(&atom->f[0][0], 0, 3 * nbytes);
        fx.lmp->force->pair->compute(0, 0);
        if (fx.lmp->force->newton) fx.lmp->comm->reverse_comm();
        out.assign(3 * nlocal, 0.0);
        for (int i = 0; i < nlocal; ++i) {
            out[3*i + 0] = atom->f[i][0];
            out[3*i + 1] = atom->f[i][1];
            out[3*i + 2] = atom->f[i][2];
        }
    };

    std::vector<double> Fp, Fm;
    displace_and_force(+eps, Fp);
    displace_and_force(-eps, Fm);
    // Restore locals + ghosts to x*.
    for (int i = 0; i < n_all; ++i) {
        atom->x[i][0] = x_save[3*i + 0];
        atom->x[i][1] = x_save[3*i + 1];
        atom->x[i][2] = x_save[3*i + 2];
    }

    Eigen::VectorXd Hv_fd = Eigen::VectorXd::Zero(3 * nlocal);
    for (int i = 0; i < 3 * nlocal; ++i) {
        Hv_fd(i) = (Fm[i] - Fp[i]) / (2.0 * eps);
    }
    Eigen::VectorXd Hv_assembled = H * v;

    // FD via large ε=2e-4 and assembled H via h=1e-4 carry independent
    // truncation noise; relative agreement to 1e-3 confirms the
    // assembler is solving the same problem PHONON would.
    const double denom = std::max(1.0, Hv_assembled.cwiseAbs().maxCoeff());
    const double rel  = (Hv_fd - Hv_assembled).cwiseAbs().maxCoeff() / denom;
    CHECK(rel < 1e-3);
}

TEST_CASE("LammpsHessianAssembler — iso box DOF is positive and "
          "consistent with energy curvature",
          "[lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());
    const int nlocal = fx.lmp->atom->nlocal;
    REQUIRE(nlocal == 4);

    tadah::hpo::LammpsHessianAssembler asm_;
    asm_.set_tikhonov(0.0);
    Eigen::MatrixXd H = asm_.assemble(fx.lmp.get(), /*box_dofs=*/1);

    REQUIRE(H.rows() == 3 * nlocal + 1);
    REQUIRE(H.cols() == 3 * nlocal + 1);

    const int iso = 3 * nlocal;
    // Iso bulk modulus row should be the dominant stiffness in the
    // matrix; positive after minimisation at the relaxed cell.
    CHECK(H(iso, iso) > 0.0);

    // Symmetry on cross terms ensured by the explicit write + the
    // symmetrise pass; assert the larger symmetry check too.
    const double asym = (H - H.transpose()).cwiseAbs().maxCoeff();
    CHECK(asym < 1e-9);
}

// Phase 3.7E regression: box-box off-diagonal block must be populated
// for box_dofs=6 (was missing pre-3.7E — the assembler only computed
// diagonal H[α,α] entries via the energy stencil and left H[α,β] = 0
// for α ≠ β). On a cubic crystal in equilibrium the (xx,yy)/(xx,zz)/
// (yy,zz) off-diagonals are V·C12 (Poisson coupling) and must be
// positive (C12 > 0 for Ta: 40.75 eV, C12 = 91 GPa, with the shipped
// potential; a potential with unphysical elasticity, such as the HPO
// examples' energies-only model, C12 = -103 GPa, fails it). Symmetric
// across the box-box block.
TEST_CASE("LammpsHessianAssembler — box-box off-diagonals are populated "
          "for box_dofs=6 on a cubic crystal (Phase 3.7E)",
          "[lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());
    const int nlocal = fx.lmp->atom->nlocal;
    REQUIRE(nlocal == 4);

    tadah::hpo::LammpsHessianAssembler asm_;
    asm_.set_tikhonov(0.0);
    Eigen::MatrixXd H = asm_.assemble(fx.lmp.get(), /*box_dofs=*/6);

    REQUIRE(H.rows() == 3 * nlocal + 6);
    REQUIRE(H.cols() == 3 * nlocal + 6);

    const int box0 = 3 * nlocal;
    /* Normal-strain diagonal (xx, yy, zz): V.C11, positive after
     * relaxation (94.69 eV with the shipped potential). */
    for (int alpha = 0; alpha < 3; ++alpha) {
        INFO("alpha = " << alpha);
        CHECK(H(box0 + alpha, box0 + alpha) > 0.0);
    }

    /* Shear diagonal (yz, xz, xy). This cell is an orthogonal box, and
     * LAMMPS ignores the tilt factors of an orthogonal box: the assembler's
     * shear strains (apply_strain sets xy, xz, yz) move nothing, and their
     * energy stencil is round-off. This is the recorded box-convention issue
     * of the analytic-gradient path (RELEASE-1.6.0-FOLLOWUPS.md, "HPO LAMMPS
     * analytic-gradient box conventions": the shears are no-ops on an
     * orthogonal box), recorded and not fixed by the owner's decision of
     * 2026-09-30, as that path is unreachable with PC_LAMMPS. Fixing it
     * changes this check.
     * Pins: the box is orthogonal, and |H(shear, shear)| < 1e-2. Measured
     * with the shipped potential: 9.95e-4 (about 1e-13 eV of energy
     * round-off over h_strain^2 = 1e-10); with the HPO examples' model
     * exactly 0. The same cell made triclinic (change_box all triclinic)
     * gives 9.155 eV, V.C44/4 for C44 = 82 GPa (the assembler applies half
     * the shear): the bound is three orders below a shear that is applied.
     * Before the fix (the case had no CTest entry then): it checked
     * H(alpha, alpha) > 0 for all six, which passed or failed on the sign of
     * the round-off: it passed with the shipped potential and failed
     * (0.0 > 0.0) with the examples' model. */
    REQUIRE(fx.lmp->domain->triclinic == 0);
    for (int alpha = 3; alpha < 6; ++alpha) {
        INFO("alpha = " << alpha);
        CHECK(std::abs(H(box0 + alpha, box0 + alpha)) < 1e-2);
    }

    /* Box-box off-diagonal block must be populated (the regression
     * gate). Pre-3.7E these were silently 0; with the virial-derivative
     * fix they pick up V·C12 > 0 for Ta. */
    const double xy_sym_diag_off = H(box0 + 0, box0 + 1);
    const double xz_sym_diag_off = H(box0 + 0, box0 + 2);
    const double yz_sym_diag_off = H(box0 + 1, box0 + 2);
    INFO("Voigt diag-diag off-diagonals (V·C12-like): "
         << "H[xx,yy]=" << xy_sym_diag_off
         << "  H[xx,zz]=" << xz_sym_diag_off
         << "  H[yy,zz]=" << yz_sym_diag_off);

    /* The three diagonal-diagonal off-diagonals (xx,yy)/(xx,zz)/(yy,zz)
     * must be NON-ZERO (the regression gate). For a cubic crystal at
     * the relaxed state they are also POSITIVE (C12 > 0 for Ta) and
     * approximately equal (cubic symmetry). */
    CHECK(std::abs(xy_sym_diag_off) > 1e-3);
    CHECK(std::abs(xz_sym_diag_off) > 1e-3);
    CHECK(std::abs(yz_sym_diag_off) > 1e-3);
    CHECK(xy_sym_diag_off > 0.0);
    CHECK(xz_sym_diag_off > 0.0);
    CHECK(yz_sym_diag_off > 0.0);

    /* Cubic symmetry: the three diagonal-diagonal off-diagonals should
     * agree to FD step-size noise (~h_strain² · |∂³E/∂ε³|). On a
     * 4-atom BCC Ta cell at h_strain=1e-5 they are equal to ~1e-7
     * relative; relax to 1e-3 to absorb FD step-size + minimisation
     * noise on smaller cells. */
    const double mean_off =
        (xy_sym_diag_off + xz_sym_diag_off + yz_sym_diag_off) / 3.0;
    CHECK(std::abs(xy_sym_diag_off - mean_off) / std::abs(mean_off) < 1e-3);
    CHECK(std::abs(xz_sym_diag_off - mean_off) / std::abs(mean_off) < 1e-3);
    CHECK(std::abs(yz_sym_diag_off - mean_off) / std::abs(mean_off) < 1e-3);

    /* Diag-shear and shear-shear cross-terms should be ~0 for cubic
     * crystals at zero stress (selection rules from cubic symmetry).
     * Bound by FD step-size noise, h_strain² · |∂³E/∂ε³| ~ 1e-3 abs. On
     * this orthogonal box they are zero for the reason above as well
     * (measured below 1e-9 with the shipped potential, and on the cell made
     * triclinic too). */
    for (int alpha = 0; alpha < 3; ++alpha) {
        for (int beta = 3; beta < 6; ++beta) {
            CHECK(std::abs(H(box0 + alpha, box0 + beta)) < 1.0);
        }
    }

    /* Box-box block symmetry. Both halves filled from independent
     * stress measurements; symmetrise pass averages — should agree. */
    const Eigen::MatrixXd box = H.block(box0, box0, 6, 6);
    const double asym = (box - box.transpose()).cwiseAbs().maxCoeff();
    CHECK(asym < 1e-9);
}

TEST_CASE("LammpsHessianAssembler — Tikhonov shifts diagonal by exactly "
          "the configured magnitude",
          "[lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());
    const int nlocal = fx.lmp->atom->nlocal;
    REQUIRE(nlocal == 4);

    tadah::hpo::LammpsHessianAssembler bare;
    bare.set_tikhonov(0.0);
    Eigen::MatrixXd H0 = bare.assemble(fx.lmp.get(), 0);

    tadah::hpo::LammpsHessianAssembler tikh;
    tikh.set_tikhonov(1e-6);
    Eigen::MatrixXd Ht = tikh.assemble(fx.lmp.get(), 0);

    Eigen::VectorXd diff_diag = Ht.diagonal() - H0.diagonal();
    // Diagonal lift is exact to FP; off-diagonals identical.
    for (int i = 0; i < diff_diag.size(); ++i) {
        CHECK(std::abs(diff_diag(i) - 1e-6) < 1e-12);
    }
    Eigen::MatrixXd offdiag_diff = (Ht - H0);
    offdiag_diff.diagonal().setZero();
    CHECK(offdiag_diff.cwiseAbs().maxCoeff() < 1e-12);
}

TEST_CASE("LammpsHessianAssembler — refreshes capture_adjoint_state "
          "snapshot at x* after the central-diff sweep destroys it",
          "[lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());
    auto* lt = fx.pair->lammps_tadah();
    REQUIRE(lt != nullptr);

    // refresh_snapshot defaults true ⇒ post-call the snapshot is valid
    // and dimensioned to atom->nlocal.
    tadah::hpo::LammpsHessianAssembler refresh;
    (void)refresh.assemble(fx.lmp.get(), 0);
    CHECK(lt->adjoint_snapshot.valid);
    CHECK(lt->adjoint_snapshot.n_atoms ==
          static_cast<std::size_t>(fx.lmp->atom->nlocal));
    CHECK_FALSE(lt->capture_adjoint_state);  // flag restored to off

    // refresh_snapshot=false leaves capture_adjoint_state cleared and
    // the snapshot not re-captured (stays in whatever state the
    // assembler's central-diff loop left it in).
    lt->adjoint_snapshot.valid = false;
    lt->adjoint_snapshot.reset(0, 0);
    tadah::hpo::LammpsHessianAssembler no_refresh;
    no_refresh.set_refresh_snapshot(false);
    (void)no_refresh.assemble(fx.lmp.get(), 0);
    CHECK_FALSE(lt->capture_adjoint_state);
    CHECK_FALSE(lt->adjoint_snapshot.valid);
}

/* LAMMPS-free smoke: ensures the assembler symbol links and its setters
 * round-trip without a LAMMPS instance. Picked up in the default test run
 * (discovered, not hidden). */
TEST_CASE("LammpsHessianAssembler — header + class symbol present "
          "without LAMMPS instance",
          "[lammps_hessian][phase3]")
{
    tadah::hpo::LammpsHessianAssembler asm_;
    asm_.set_tikhonov(1.0e-7);
    asm_.set_symmetrize(true);
    asm_.set_refresh_snapshot(false);
    SUCCEED("assembler header compiles and class instantiates");
}

/* Box-DOFs argument validation can run without a LAMMPS instance — the
 * `box_dofs ∈ {0,1,6}` precondition fires before any pointer
 * dereference. Catches accidental changes to the public API. */
TEST_CASE("LammpsHessianAssembler — rejects out-of-range box_dofs",
          "[lammps_hessian][phase3]")
{
    tadah::hpo::LammpsHessianAssembler asm_;
    /* Null lmp would trip require_pair_tadah after the box_dofs check —
     * but the box_dofs check fires first, so we can pass nullptr safely
     * to confirm the validation path. */
    CHECK_THROWS_AS(asm_.assemble(nullptr, 2),  std::runtime_error);
    CHECK_THROWS_AS(asm_.assemble(nullptr, -1), std::runtime_error);
    CHECK_THROWS_AS(asm_.assemble(nullptr, 7),  std::runtime_error);
}

/* Phase 3.7E follow-up: profile the vflag=1 overhead added to
 * LammpsHessianAssembler's box-DOF sweep. 3.7E flipped vflag from 0 to 1
 * on the 12 strain perturbations so pair->virial gets populated and the
 * off-diagonal box-box block can be assembled from -∂virial[α]/∂ε_β
 * (reusing the same 12 LAMMPS computes). The added cost per compute is
 * the per-atom virial accumulation inside pair_tadah's loops; the post-
 * compute virial[] copy is 6 doubles and negligible by inspection.
 *
 * The bench runs two BENCHMARK blocks side by side on the same relaxed
 * BCC Ta state: vflag=0 (pre-3.7E) and vflag=1 (post-3.7E). Their ratio
 * = per-compute overhead of 3.7E's change. The total assembler cost
 * picks this up on 12 of the 12 + 6·nlocal compute calls (atom-block
 * sweep keeps vflag=0). For the production HPO workload (~16-atom
 * cells, box_dofs=1 or 6) the box-DOF computes are O(12); atom-DOF
 * computes are O(6·16)=96; vflag overhead applies to ~11% of computes.
 *
 * Memory predicts <1% total assembler overhead. The bench measures the
 * per-compute multiplier directly so future regressions show up
 * immediately. */
TEST_CASE("LammpsHessianAssembler — vflag=1 overhead microbench "
          "(Phase 3.7E follow-up)",
          "[lammps_hessian_perf][lammps_hessian][phase3][.][!benchmark]")
{
    LammpsFixture fx(hpo_test::bench_pot_path());

    LAMMPS_NS::Pair* pair = fx.lmp->force->pair;
    LAMMPS_NS::Atom* atom = fx.lmp->atom;
    REQUIRE(pair != nullptr);

    /* Mirror compute_clear_and_eval's body so the bench measures the
     * actual per-perturbation work the Hessian assembler does. */
    auto clear_and_compute = [&](int eflag, int vflag) {
        fx.lmp->neighbor->ago = 0;
        const std::size_t nbytes_local = sizeof(double) * atom->nlocal;
        std::size_t nbytes = nbytes_local;
        if (fx.lmp->force->newton)
            nbytes += sizeof(double) * atom->nghost;
        if (nbytes) std::memset(&atom->f[0][0], 0, 3 * nbytes);
        pair->compute(eflag, vflag);
        if (fx.lmp->force->newton) fx.lmp->comm->reverse_comm();
    };

    /* Warmup: a single compute to populate caches before the BENCHMARK
     * timing loop kicks in. */
    clear_and_compute(/*eflag=*/1, /*vflag=*/0);

    BENCHMARK("pair_tadah::compute(eflag=1, vflag=0) — pre-3.7E baseline")
    {
        clear_and_compute(1, 0);
        return pair->eng_vdwl;
    };

    BENCHMARK("pair_tadah::compute(eflag=1, vflag=1) — 3.7E current")
    {
        clear_and_compute(1, 1);
        return pair->eng_vdwl + pair->virial[0];
    };
}

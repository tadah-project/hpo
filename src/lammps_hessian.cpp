#include <tadah/hpo/lammps_hessian.h>

#include <lammps.h>
#include <atom.h>
#include <comm.h>
#include <domain.h>
#include <force.h>
#include <neighbor.h>
#include <pair.h>

#include <tadah/tadah.lammps/lammps_tadah.h>
#include <pair_tadah.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tadah::hpo {

namespace {

/* The six Voigt strain modes the assembler can apply, as tensors. Each
 * column alpha gives the symmetric eps_ij that turns Voigt strain alpha into a
 * deformation gradient F = I + eps. Engineering shear (Voigt 3..5) is
 * applied with a factor of 1/2 on the off-diagonal entries -- the matched
 * lattice-mechanics convention. */
struct StrainBasis {
    double e[3][3];
};

inline StrainBasis voigt_basis(int alpha) {
    StrainBasis s{};
    switch (alpha) {
        case 0: s.e[0][0] = 1.0; break;                                // xx
        case 1: s.e[1][1] = 1.0; break;                                // yy
        case 2: s.e[2][2] = 1.0; break;                                // zz
        case 3: s.e[1][2] = s.e[2][1] = 0.5; break;                    // yz
        case 4: s.e[0][2] = s.e[2][0] = 0.5; break;                    // xz
        case 5: s.e[0][1] = s.e[1][0] = 0.5; break;                    // xy
        default: throw std::runtime_error(
            "LammpsHessianAssembler: voigt index out of range");
    }
    return s;
}

inline LAMMPS_NS::PairTadah* require_pair_tadah(LAMMPS_NS::LAMMPS* lmp) {
    if (!lmp || !lmp->force || !lmp->force->pair) {
        throw std::runtime_error(
            "LammpsHessianAssembler: LAMMPS instance has no pair style");
    }
    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    if (!pt) {
        throw std::runtime_error(
            "LammpsHessianAssembler: active pair style is not pair tadah");
    }
    return pt;
}

/* LAMMPS keeps three pieces of box state that must be edited together
 * when applying a strain: `boxlo/boxhi`, the three tilts `xy/xz/yz`, and
 * `domain->h` (Voigt cell matrix). Saving everything here lets the
 * caller restore byte-identical state after the central-diff sweep. */
struct BoxSnapshot {
    double boxlo[3];
    double boxhi[3];
    double xy{}, xz{}, yz{};
    double h[6]{};
    double h_inv[6]{};
};

inline BoxSnapshot snapshot_box(LAMMPS_NS::Domain* dom) {
    BoxSnapshot s{};
    std::memcpy(s.boxlo, dom->boxlo, sizeof(s.boxlo));
    std::memcpy(s.boxhi, dom->boxhi, sizeof(s.boxhi));
    s.xy = dom->xy;  s.xz = dom->xz;  s.yz = dom->yz;
    std::memcpy(s.h,     dom->h,     sizeof(s.h));
    std::memcpy(s.h_inv, dom->h_inv, sizeof(s.h_inv));
    return s;
}

inline void restore_box(LAMMPS_NS::Domain* dom, const BoxSnapshot& s) {
    std::memcpy(dom->boxlo, s.boxlo, sizeof(s.boxlo));
    std::memcpy(dom->boxhi, s.boxhi, sizeof(s.boxhi));
    dom->xy = s.xy;  dom->xz = s.xz;  dom->yz = s.yz;
    std::memcpy(dom->h,     s.h,     sizeof(s.h));
    std::memcpy(dom->h_inv, s.h_inv, sizeof(s.h_inv));
    dom->set_global_box();
    dom->set_local_box();
}

/* Apply F = I + eps * scale to the saved box state and write it into
 * `dom`. Atoms held in lamda coords by the caller scale affinely on the
 * trailing lamda2x. */
inline void apply_strain(LAMMPS_NS::Domain* dom,
                         const BoxSnapshot& base,
                         const StrainBasis& eps,
                         double scale)
{
    /* Reconstruct the saved 3x3 box edge matrix B, then push through
     * F.B. LAMMPS' upper-triangular convention: a=(xprd,0,0),
     * b=(xy,yprd,0), c=(xz,yz,zprd). Restoring it after F.B requires
     * the columns to stay upper-triangular -- true for diagonal strains
     * but only approximately true for shear modes at the scales we
     * use (<=1e-5). The residual off-triangular pieces are dropped; the
     * resulting drift is O(scale^2) and bounded by the central-diff
     * step-size noise of the Hessian itself. */
    const double xprd = base.boxhi[0] - base.boxlo[0];
    const double yprd = base.boxhi[1] - base.boxlo[1];
    const double zprd = base.boxhi[2] - base.boxlo[2];

    double B[3][3] = {
        { xprd, base.xy, base.xz },
        { 0.0,  yprd,    base.yz },
        { 0.0,  0.0,     zprd    }
    };

    double F[3][3] = {
        { 1.0 + scale * eps.e[0][0],       scale * eps.e[0][1], scale * eps.e[0][2] },
        {       scale * eps.e[1][0], 1.0 + scale * eps.e[1][1], scale * eps.e[1][2] },
        {       scale * eps.e[2][0],       scale * eps.e[2][1], 1.0 + scale * eps.e[2][2] }
    };

    double FB[3][3]{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                FB[i][j] += F[i][k] * B[k][j];

    const double new_xprd = FB[0][0];
    const double new_yprd = FB[1][1];
    const double new_zprd = FB[2][2];
    const double new_xy   = FB[0][1];
    const double new_xz   = FB[0][2];
    const double new_yz   = FB[1][2];

    dom->boxlo[0] = base.boxlo[0];
    dom->boxlo[1] = base.boxlo[1];
    dom->boxlo[2] = base.boxlo[2];
    dom->boxhi[0] = base.boxlo[0] + new_xprd;
    dom->boxhi[1] = base.boxlo[1] + new_yprd;
    dom->boxhi[2] = base.boxlo[2] + new_zprd;
    dom->xy = new_xy;  dom->xz = new_xz;  dom->yz = new_yz;
    dom->set_global_box();
    dom->set_local_box();
}

inline void compute_clear_and_eval(LAMMPS_NS::LAMMPS* lmp,
                                   LAMMPS_NS::Pair*  pair,
                                   int eflag, int vflag)
{
    /* Mirror DynamicalMatrix::update_force()'s minimum sequence:
     * neighbor->ago=0 to keep stencils current, clear forces on local
     * and ghost atoms, then drive the pair compute. Newton-pair
     * reverse_comm is a no-op in serial; included for parity with the
     * production path. */
    lmp->neighbor->ago = 0;
    LAMMPS_NS::Atom* atom = lmp->atom;
    const std::size_t nbytes_local = sizeof(double) * atom->nlocal;
    std::size_t nbytes = nbytes_local;
    if (lmp->force->newton) nbytes += sizeof(double) * atom->nghost;
    if (nbytes) std::memset(&atom->f[0][0], 0, 3 * nbytes);

    pair->compute(eflag, vflag);

    if (lmp->force->newton) lmp->comm->reverse_comm();
}

/* Pull post-compute force on local atoms into a contiguous buffer. */
inline void copy_local_forces(LAMMPS_NS::Atom* atom,
                              std::vector<double>& out)
{
    const int nlocal = atom->nlocal;
    out.assign(static_cast<std::size_t>(3 * nlocal), 0.0);
    double** f = atom->f;
    for (int i = 0; i < nlocal; ++i) {
        out[3*i + 0] = f[i][0];
        out[3*i + 1] = f[i][1];
        out[3*i + 2] = f[i][2];
    }
}

}  // anonymous namespace

Eigen::MatrixXd
LammpsHessianAssembler::assemble(LAMMPS_NS::LAMMPS* lmp,
                                 int    box_dofs,
                                 double h_atom,
                                 double h_strain)
{
    if (box_dofs != 0 && box_dofs != 1 && box_dofs != 6) {
        throw std::runtime_error(
            "LammpsHessianAssembler: box_dofs must be 0, 1, or 6");
    }
    if (h_atom   <= 0.0) throw std::runtime_error("h_atom must be > 0");
    if (h_strain <= 0.0) throw std::runtime_error("h_strain must be > 0");

    LAMMPS_NS::PairTadah* pair = require_pair_tadah(lmp);
    LAMMPS_NS::Atom*   atom = lmp->atom;
    LAMMPS_NS::Domain* dom  = lmp->domain;

    auto* lt = pair->lammps_tadah();
    const bool flag_save = lt ? lt->capture_adjoint_state : false;
    if (lt) lt->capture_adjoint_state = false;

    const int nlocal      = atom->nlocal;
    const int n_atom_dof  = 3 * nlocal;
    const int n           = n_atom_dof + box_dofs;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);

    const int nghost      = atom->nghost;
    const int n_all       = nlocal + nghost;
    int* sametag          = atom->sametag;  // may be nullptr (atom_modify map none)

    /* Save x* state for every owned + ghost atom. Ghosts must be tracked
     * so that the sametag-chain displacement and the box-DOF strain (both
     * of which touch ghost positions) can be undone byte-for-byte.
     * atom->x is double**: row pointers may be reused by LAMMPS internal
     * storage; copy by value into a flat buffer that we also write back
     * row-by-row on restore. */
    std::vector<double> x_save(static_cast<std::size_t>(3 * n_all), 0.0);
    {
        double** x = atom->x;
        for (int i = 0; i < n_all; ++i) {
            x_save[3*i + 0] = x[i][0];
            x_save[3*i + 1] = x[i][1];
            x_save[3*i + 2] = x[i][2];
        }
    }
    BoxSnapshot box_save = snapshot_box(dom);

    std::vector<double> F_plus, F_minus;
    F_plus.reserve(static_cast<std::size_t>(3 * nlocal));
    F_minus.reserve(static_cast<std::size_t>(3 * nlocal));

    /* Atom DOF sweep - central diff of forces under per-atom displacements.
     * H[ik, jl] = -dF_j^l/dx_i^k = (F_-(x-h.e_ik)[j,l] - F_+(x+h.e_ik)[j,l]) / (2h).
     *
     * Sametag walk: in a periodic system every local atom has one or
     * more ghost images that pair_tadah's neighbor loop iterates over.
     * Displacing only the local index leaves ghosts at the old cartesian
     * position, so atoms that see the ghost as their neighbor never feel
     * the displacement -- H ends up wrong (asymmetric + spurious acoustic
     * stiffness). PHONON's DynamicalMatrix::displace_atom solves this by
     * walking atom->sametag, which threads each local atom through its
     * ghost copies. Same fix applies here. */
    auto displace_with_ghosts = [&](int local_idx, int axis, double delta) {
        atom->x[local_idx][axis] += delta;
        if (!sametag) return;
        int j = local_idx;
        while (sametag[j] >= 0) {
            j = sametag[j];
            atom->x[j][axis] += delta;
        }
    };
    auto restore_atom_axis = [&](int local_idx, int axis) {
        atom->x[local_idx][axis] = x_save[3*local_idx + axis];
        if (!sametag) return;
        int j = local_idx;
        while (sametag[j] >= 0) {
            j = sametag[j];
            atom->x[j][axis] = x_save[3*j + axis];
        }
    };
    for (int i = 0; i < nlocal; ++i) {
        for (int k = 0; k < 3; ++k) {
            const int row = 3*i + k;

            displace_with_ghosts(i, k, +h_atom);
            compute_clear_and_eval(lmp, pair, /*eflag=*/0, /*vflag=*/0);
            copy_local_forces(atom, F_plus);
            restore_atom_axis(i, k);

            displace_with_ghosts(i, k, -h_atom);
            compute_clear_and_eval(lmp, pair, 0, 0);
            copy_local_forces(atom, F_minus);
            restore_atom_axis(i, k);

            const double inv2h = 1.0 / (2.0 * h_atom);
            for (int j = 0; j < nlocal; ++j) {
                for (int l = 0; l < 3; ++l) {
                    const int col = 3*j + l;
                    H(row, col) = (F_minus[3*j + l] - F_plus[3*j + l]) * inv2h;
                }
            }
        }
    }

    /* Box DOF sweep. Voigt order eps_xx, eps_yy, eps_zz, eps_yz, eps_xz, eps_xy. For
     * box_dofs=1 we treat that single DOF as iso strain eps_iso applied as
     * eps_ij = delta_ij.eps_iso (i.e. F = (1+eps).I).
     *
     * For each box DOF alpha the strain mode displaces atoms affinely (held
     * in lamda coords across the box edit). H[i*3+k, n_atom_dof + alpha] is
     * read from the per-atom forces at strain +/-h_strain; the diagonal
     * H[n_atom_dof + alpha, n_atom_dof + alpha] is read from total pair energy
     * via E(+h_s) - 2.E(0) + E(-h_s).
     *
     * Phase 3.7E: off-diagonal box-box terms H[alpha, beta] for alpha!=beta read from
     * the LAMMPS pair virial (d^2E/deps_alphadeps_beta = -dvirial[alpha]/deps_beta). Mass
     * conserved: vflag flipped to 1 so virial gets populated; the
     * existing 12 LAMMPS computes are reused (no extra strain
     * perturbations).
     *
     * LAMMPS pair->virial layout is {xx, yy, zz, xy, xz, yz}, but our
     * Voigt convention here is {xx, yy, zz, yz, xz, xy} so the off-
     * diagonal slots need a permutation when reading. */
    if (box_dofs > 0) {
        /* Single-atom guard: a single atom has no internal Hessian rows
         * to mix with strain; the cross/diagonal terms are still valid. */
        const auto n_strain = static_cast<int>(box_dofs);

        /* Need E(0). Recompute at saved state with energy flag. */
        compute_clear_and_eval(lmp, pair, /*eflag=*/1, /*vflag=*/0);
        const double E0 = lmp->force->pair->eng_vdwl + lmp->force->pair->eng_coul;

        std::vector<double> F_pe(static_cast<std::size_t>(3 * nlocal), 0.0);
        std::vector<double> F_me(static_cast<std::size_t>(3 * nlocal), 0.0);

        /* Saved virials at +eps_beta / -eps_beta for each beta, indexed by Voigt alpha
         * (post-permutation). Used after the alpha loop to assemble the
         * off-diagonal box-box block. Flat row-major: [beta . 6 + v]. */
        std::vector<double> virial_p(
            static_cast<std::size_t>(n_strain) * 6u, 0.0);
        std::vector<double> virial_m(
            static_cast<std::size_t>(n_strain) * 6u, 0.0);

        /* LAMMPS virial Voigt order -> our Voigt order:
         *   our[0..2] = LAMMPS[0..2]   (xx, yy, zz)
         *   our[3]    = LAMMPS[5]      (yz)
         *   our[4]    = LAMMPS[4]      (xz)
         *   our[5]    = LAMMPS[3]      (xy) */
        constexpr int virial_perm[6] = { 0, 1, 2, 5, 4, 3 };

        for (int alpha = 0; alpha < n_strain; ++alpha) {
            const StrainBasis eps = (box_dofs == 1)
                ? StrainBasis{ {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}} } // iso
                : voigt_basis(alpha);

            /* +h_strain. Convert ALL atoms (locals + ghosts) to lamda so
             * ghost periodic images scale affinely with the box; otherwise
             * pair_tadah's neighbor loop sees ghost neighbours at the
             * unstrained cartesian position and the energy/force curvature
             * along eps comes out attenuated and asymmetric.
             *
             * vflag=1 to populate pair->virial (energy + virial; per-atom
             * virial not needed). */
            dom->x2lamda(n_all);
            apply_strain(dom, box_save, eps, +h_strain);
            dom->lamda2x(n_all);
            compute_clear_and_eval(lmp, pair, 1, 1);
            const double E_p = lmp->force->pair->eng_vdwl + lmp->force->pair->eng_coul;
            copy_local_forces(atom, F_pe);
            for (int v = 0; v < 6; ++v) {
                virial_p[static_cast<std::size_t>(alpha) * 6u
                       + static_cast<std::size_t>(v)] =
                    lmp->force->pair->virial[v];
            }

            /* Restore positions for both locals AND ghosts; the lamda
             * round-trip on the unstrained box would otherwise re-derive
             * cartesians from already-strained lamdas. Explicit overwrite
             * is cheaper and matches the save buffer's coverage. */
            for (int i = 0; i < n_all; ++i) {
                atom->x[i][0] = x_save[3*i + 0];
                atom->x[i][1] = x_save[3*i + 1];
                atom->x[i][2] = x_save[3*i + 2];
            }
            restore_box(dom, box_save);

            /* -h_strain */
            dom->x2lamda(n_all);
            apply_strain(dom, box_save, eps, -h_strain);
            dom->lamda2x(n_all);
            compute_clear_and_eval(lmp, pair, 1, 1);
            const double E_m = lmp->force->pair->eng_vdwl + lmp->force->pair->eng_coul;
            copy_local_forces(atom, F_me);
            for (int v = 0; v < 6; ++v) {
                virial_m[static_cast<std::size_t>(alpha) * 6u
                       + static_cast<std::size_t>(v)] =
                    lmp->force->pair->virial[v];
            }

            for (int i = 0; i < n_all; ++i) {
                atom->x[i][0] = x_save[3*i + 0];
                atom->x[i][1] = x_save[3*i + 1];
                atom->x[i][2] = x_save[3*i + 2];
            }
            restore_box(dom, box_save);

            /* Cross terms H[ik, alpha] = -dF_i^k/deps_alpha.
             * Symmetric counterpart H[alpha, ik] written from the same
             * central difference; we average via symmetrise below. */
            const int col_alpha = n_atom_dof + alpha;
            const double inv2hs = 1.0 / (2.0 * h_strain);
            for (int i = 0; i < nlocal; ++i) {
                for (int k = 0; k < 3; ++k) {
                    const double d = (F_me[3*i + k] - F_pe[3*i + k]) * inv2hs;
                    H(3*i + k, col_alpha) = d;
                    H(col_alpha, 3*i + k) = d;
                }
            }

            /* Diagonal H[alpha, alpha] = d^2E/deps_alpha^2 via 3-point energy stencil. */
            H(col_alpha, col_alpha) =
                (E_p - 2.0 * E0 + E_m) / (h_strain * h_strain);
        }

        /* Box-box off-diagonal block (Phase 3.7E). For alpha!=beta:
         *
         *   H[alpha, beta] = d^2E/deps_alphadeps_beta = -dvirial[alpha]/deps_beta
         *
         * LAMMPS' pair->virial[alpha] = -V.sigma_alpha (compressive-positive
         * convention: virial[alpha] > 0 under compression). Thus
         * dE/deps_alpha = V.sigma_alpha = -virial[alpha], and the second derivative is
         *   H[alpha, beta] = -(virial_perm(alpha)(+eps_beta) - virial_perm(alpha)(-eps_beta)) / (2h).
         *
         * Both H[alpha, beta] and H[beta, alpha] computed independently from
         * different perturbation directions; the trailing symmetrise
         * pass (H = 1/2(H + H^T)) averages them, picking up off-diagonal
         * stress-stress symmetry to FP. Skip iso (n_strain == 1: no
         * box-box off-diagonals to populate). */
        if (n_strain > 1) {
            const double inv2hs = 1.0 / (2.0 * h_strain);
            for (int alpha = 0; alpha < n_strain; ++alpha) {
                const int va = virial_perm[alpha];
                for (int beta = 0; beta < n_strain; ++beta) {
                    if (alpha == beta) continue;
                    const double dv =
                        virial_p[static_cast<std::size_t>(beta) * 6u
                               + static_cast<std::size_t>(va)]
                      - virial_m[static_cast<std::size_t>(beta) * 6u
                               + static_cast<std::size_t>(va)];
                    H(n_atom_dof + alpha, n_atom_dof + beta) = -dv * inv2hs;
                }
            }
        }
        (void)E0;  // referenced only inside the loop; silence -Wunused.
    }

    /* Final state restoration. compute_clear_and_eval after every step
     * leaves atom->f populated with the last strained state; the
     * snapshot refresh below overwrites it, but the position copy here
     * is the load-bearing invariant. Ghosts must also be restored -- the
     * sametag walks in the atom sweep and the lamda2x(n_all) in the box
     * sweep both write past nlocal. */
    {
        double** x = atom->x;
        for (int i = 0; i < n_all; ++i) {
            x[i][0] = x_save[3*i + 0];
            x[i][1] = x_save[3*i + 1];
            x[i][2] = x_save[3*i + 2];
        }
    }
    restore_box(dom, box_save);

    if (symmetrize_) {
        /* Aliasing-safe symmetrise. `H = 0.5 * (H + H.transpose())`
         * evaluates lazily -- by the time the (j, i) entry is written,
         * the (i, j) entry has already been overwritten. Force a
         * temporary via `.eval()` so the rhs computes from the
         * original H values throughout. Pre-3.7E this surfaced as
         * ~1e-8 asymmetry on the Voigt-6 box block, masked at the
         * iso path (1x1 box block trivially symmetric). */
        H = (0.5 * (H + H.transpose())).eval();
    }
    if (tikhonov_ > 0.0) {
        H.diagonal().array() += tikhonov_;
    }

    if (refresh_snapshot_ && lt) {
        lt->capture_adjoint_state = true;
        compute_clear_and_eval(lmp, pair, /*eflag=*/0, /*vflag=*/0);
        lt->capture_adjoint_state = flag_save;
    } else if (lt) {
        lt->capture_adjoint_state = flag_save;
        /* Restore atom->f to a meaningful state at x*. */
        compute_clear_and_eval(lmp, pair, 0, 0);
    } else {
        compute_clear_and_eval(lmp, pair, 0, 0);
    }

    return H;
}

}  // namespace tadah::hpo

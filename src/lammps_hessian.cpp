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

#include <array>
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

/* 3x3 matrices of the box block (row-major). */
using Mat3 = std::array<std::array<double, 3>, 3>;

/* The edge matrix of a LAMMPS box from its h (xprd, yprd, zprd, yz, xz,
 * xy): columns a = (xprd, 0, 0), b = (xy, yprd, 0), c = (xz, yz, zprd), so
 * that x = B lamda + boxlo (Domain::lamda2x). For an orthogonal box LAMMPS
 * leaves h[3..5] at 0 whatever xy, xz, yz hold. */
inline Mat3 box_edges(const double* h)
{
    return {{ {{ h[0], h[5], h[4] }},
              {{ 0.0,  h[1], h[3] }},
              {{ 0.0,  0.0,  h[2] }} }};
}

inline Mat3 mat3_mul(const Mat3& a, const Mat3& b)
{
    Mat3 c{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                c[i][j] += a[i][k] * b[k][j];
    return c;
}

inline Mat3 mat3_inverse(const Mat3& F)
{
    const double det =
        F[0][0] * (F[1][1] * F[2][2] - F[1][2] * F[2][1])
      - F[0][1] * (F[1][0] * F[2][2] - F[1][2] * F[2][0])
      + F[0][2] * (F[1][0] * F[2][1] - F[1][1] * F[2][0]);
    Mat3 inv{};
    inv[0][0] =  (F[1][1] * F[2][2] - F[1][2] * F[2][1]) / det;
    inv[0][1] = -(F[0][1] * F[2][2] - F[0][2] * F[2][1]) / det;
    inv[0][2] =  (F[0][1] * F[1][2] - F[0][2] * F[1][1]) / det;
    inv[1][0] = -(F[1][0] * F[2][2] - F[1][2] * F[2][0]) / det;
    inv[1][1] =  (F[0][0] * F[2][2] - F[0][2] * F[2][0]) / det;
    inv[1][2] = -(F[0][0] * F[1][2] - F[0][2] * F[1][0]) / det;
    inv[2][0] =  (F[1][0] * F[2][1] - F[1][1] * F[2][0]) / det;
    inv[2][1] = -(F[0][0] * F[2][1] - F[0][1] * F[2][0]) / det;
    inv[2][2] =  (F[0][0] * F[1][1] - F[0][1] * F[1][0]) / det;
    return inv;
}

/* dE/ds_alpha at a homogeneously strained state of the box, from the pair
 * virial of that state. The atoms sit at x = F x0 (F = B B0^-1, B the
 * strained box's edges and B0 the saved box's; the assembler holds the
 * atoms, ghosts included, in lamda coordinates across the box edit), and
 * mode alpha moves them along d_alpha x0 (d_alpha = dF/ds_alpha), so
 *   dE/ds_alpha = sum_i dE/dx_i . (d_alpha x0_i) = -tr(d_alpha F^-1 W)
 * with W_ca = sum_i x_ic f_ia, the virial LAMMPS accumulates (pair->virial,
 * order xx, yy, zz, xy, xz, yz; symmetric). At F = I and d = eps_xx this is
 * -virial[0]; for the iso mode (d = I) -(virial[0] + virial[1] +
 * virial[2]). */
inline double strain_derivative(const Mat3& d_alpha, const Mat3& F,
                                const double* virial)
{
    const double W[3][3] = {
        { virial[0], virial[3], virial[4] },
        { virial[3], virial[1], virial[5] },
        { virial[4], virial[5], virial[2] }
    };
    const Mat3 Finv = mat3_inverse(F);
    double tr = 0.0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
                tr += d_alpha[a][b] * Finv[b][c] * W[c][a];
    return -tr;
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
     * read from the per-atom forces at strain +/-h_strain.
     *
     * The box-box block H[alpha, beta] = d^2E/ds_alpha ds_beta, along the
     * deformations the modes apply (the paths of the cross terms), is the
     * central difference over s_beta of dE/ds_alpha, which
     * strain_derivative() reads from the LAMMPS pair virial of each strained
     * state (-tr(d_alpha F^-1 W), F and d_alpha from the strained boxes as
     * LAMMPS holds them): the diagonal and the off-diagonal (Phase 3.7E)
     * alike, from the 2 * box_dofs strained computes the cross terms need
     * (vflag=1, no extra strain perturbations). The virial is a first
     * derivative, so its difference over 2 h_strain carries the virial's
     * round-off over h_strain. The diagonal used to be the energy's second
     * difference (E(+h) - 2 E(0) + E(-h)) / h^2 along the same path, which
     * multiplies the energy's round-off by 1/h^2 = 1e10: about 1e-11 eV of
     * energy noise moved the iso entry of a 16-atom Ta cell by 1e-4
     * relative (3393.8 eV against 3394.35 eV). The off-diagonal used
     * -dvirial[alpha]/deps_beta in the Voigt convention, the same for two
     * normal modes, and without the F^-1 term and the applied shear
     * otherwise.
     *
     * LAMMPS pair->virial layout is {xx, yy, zz, xy, xz, yz}, but our
     * Voigt convention here is {xx, yy, zz, yz, xz, xy}; strain_derivative()
     * reads the tensor whole. */
    if (box_dofs > 0) {
        /* Single-atom guard: a single atom has no internal Hessian rows
         * to mix with strain; the cross/diagonal terms are still valid. */
        const auto n_strain = static_cast<int>(box_dofs);

        std::vector<double> F_pe(static_cast<std::size_t>(3 * nlocal), 0.0);
        std::vector<double> F_me(static_cast<std::size_t>(3 * nlocal), 0.0);

        /* Saved virials at +eps_beta / -eps_beta for each beta, in LAMMPS'
         * order (xx, yy, zz, xy, xz, yz). Used after the alpha loop to
         * assemble the box-box block. Flat row-major: [beta . 6 + v]. */
        std::vector<double> virial_p(
            static_cast<std::size_t>(n_strain) * 6u, 0.0);
        std::vector<double> virial_m(
            static_cast<std::size_t>(n_strain) * 6u, 0.0);
        /* The edges of each strained box as LAMMPS holds them (an
         * orthogonal box ignores apply_strain's tilts). */
        std::vector<Mat3> box_p(static_cast<std::size_t>(n_strain));
        std::vector<Mat3> box_m(static_cast<std::size_t>(n_strain));

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
            box_p[static_cast<std::size_t>(alpha)] = box_edges(dom->h);
            compute_clear_and_eval(lmp, pair, 1, 1);
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
            box_m[static_cast<std::size_t>(alpha)] = box_edges(dom->h);
            compute_clear_and_eval(lmp, pair, 1, 1);
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
        }

        /* Box-box block: H[alpha, beta] is the central difference over
         * s_beta of dE/ds_alpha (strain_derivative), the diagonal and the
         * off-diagonal alike, along the deformation each mode applies: F =
         * B B0^-1 of the strained box, and d_alpha = (B_alpha(+h) -
         * B_alpha(-h)) / 2h . B0^-1 (eps_alpha for the iso and normal
         * modes; for a shear, what apply_strain's upper triangle and LAMMPS
         * make of it). Both halves come from different perturbation
         * directions; the trailing symmetrise pass (H = 1/2(H + H^T))
         * averages them. */
        const double inv2hs = 1.0 / (2.0 * h_strain);
        const Mat3 B0_inv = mat3_inverse(box_edges(box_save.h));
        std::vector<Mat3> d(static_cast<std::size_t>(n_strain));
        std::vector<Mat3> F_p(static_cast<std::size_t>(n_strain));
        std::vector<Mat3> F_m(static_cast<std::size_t>(n_strain));
        for (std::size_t m = 0; m < static_cast<std::size_t>(n_strain); ++m) {
            F_p[m] = mat3_mul(box_p[m], B0_inv);
            F_m[m] = mat3_mul(box_m[m], B0_inv);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    d[m][i][j] = (F_p[m][i][j] - F_m[m][i][j]) * inv2hs;
        }
        for (int beta = 0; beta < n_strain; ++beta) {
            const auto b = static_cast<std::size_t>(beta);
            const double* vp = &virial_p[b * 6u];
            const double* vm = &virial_m[b * 6u];
            for (int alpha = 0; alpha < n_strain; ++alpha) {
                const auto a = static_cast<std::size_t>(alpha);
                H(n_atom_dof + alpha, n_atom_dof + beta) =
                    (strain_derivative(d[a], F_p[b], vp) -
                     strain_derivative(d[a], F_m[b], vm)) * inv2hs;
            }
        }
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

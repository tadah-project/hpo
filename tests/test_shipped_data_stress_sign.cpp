// test_shipped_data_stress_sign.cpp
//
// The stress sign and the energy of the two VASP-derived datasets the HPO
// worked examples train and validate on (examples/common/data, found through
// TADAH_HPO_EXAMPLES_DIR, as in test_example_data_cells.cpp).
//
// A Tadah dataset stores the Cauchy stress, positive under tension, in
// eV/A^3: sigma = (1/V) dE/d(strain), what the model predicts. Its
// consequences for the stored energies, checked here without a model:
//   - hydrostatic strain (eos_bcc.tadahdb, cubic 2-atom cells, F = l I):
//     dE = V tr(sigma) dl/l and dV = 3 V dl/l, so tr(sigma)/3 = dE/dV;
//   - uniaxial strain eps of the reference cell a0 I along d (the rows of
//     elastic.tadahdb's cells: F = I + eps e_d e_d): the first Piola-Kirchhoff
//     stress P = J sigma F^-T has P_dd = sigma_dd, so (1/V0) dE/d(eps) =
//     sigma_dd at that strain, V0 the volume of the reference cell;
//   - simple shear: lattice vector a_r gains gamma a0 along c, r != c
//     (F = I + gamma e_c e_r, J = 1): (1/V0) dE/d(gamma) = sigma_rc.
// The derivatives are three-point central differences on the (non-uniform)
// grid of the stored strains: the slope at the middle point of the parabola
// through three neighbours, exact for a quadratic.
//
// Both files are converted from VASP OUTCARs (Datastore/Ta/DFT_DATA2/
// vasp_data: E_vs_V/bcc_reduced/*/OUTCAR, ELASTIC/archive_original_P0/OUTCAR;
// ISMEAR 1, SIGMA 0.2). The stored energy is the free energy F (the ionic
// step's "free  energy   TOTEN", the energy whose strain derivative VASP's
// stress is) plus 3.69807734 eV/atom (minus the free energy of the isolated,
// spin-polarised Ta atom). The reference values below were computed with
// Python from those OUTCARs, outside the code under test.
//
// Before the fix both files hold what the 1.0.0-1.6.0 OUTCAR reader stored:
// VASP's stress as printed ("in kB" times 6.241509074e-4, positive under
// compression, i.e. -sigma) and the energy without entropy E_wo instead of F
// (RELEASE-1.6.0-FOLLOWUPS.md, known-issues decisions, D;
// tmp/known-issues/report_stress.md section 4, report_readers.md A.6).

#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

StructureDB load_example_data(const std::string& file)
{
    const fs::path path =
        fs::path(TADAH_HPO_EXAMPLES_DIR) / "common" / "data" / file;
    INFO("dataset " << path.string());
    REQUIRE(fs::exists(path));
    StructureDB stdb;
    stdb.add(path.string());
    return stdb;
}

/* |a0 . (a1 x a2)| of the rows, written out (not Structure::get_volume). */
double cell_volume(const Structure& st)
{
    const auto& c = st.cell;
    return std::fabs(c(0, 0) * (c(1, 1) * c(2, 2) - c(1, 2) * c(2, 1))
                   - c(0, 1) * (c(1, 0) * c(2, 2) - c(1, 2) * c(2, 0))
                   + c(0, 2) * (c(1, 0) * c(2, 1) - c(1, 1) * c(2, 0)));
}

bool force_free(const Structure& st)
{
    for (const auto& a : st.atoms)
        for (int k = 0; k < 3; ++k)
            if (a.force(k) != 0.0) return false;
    return true;
}

/* Every cell entry equals rows[i][j] within 1e-9 A. */
bool cell_is(const Structure& st, const double (&rows)[3][3])
{
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (std::fabs(st.cell(i, j) - rows[i][j]) > 1e-9) return false;
    return true;
}

/* dE/dx at x0 from (xm, em), (x0, e0), (xp, ep), xm < x0 < xp: the slope at
 * x0 of the parabola through the three points (second-order accurate on a
 * non-uniform grid, exact for a quadratic). */
double central_derivative(double xm, double x0, double xp,
                          double em, double e0, double ep)
{
    const double h1 = x0 - xm;
    const double h2 = xp - x0;
    return (h1 * h1 * ep - h2 * h2 * em + (h2 * h2 - h1 * h1) * e0)
         / (h1 * h2 * (h1 + h2));
}

}  // namespace

/* Pins: eos_bcc.tadahdb stores tr(sigma)/3 = dE/dV. Least squares over the
 * 135 interior points of the 137 distinct volumes (a3.32 and MINIMA/a3.32
 * are one calculation, stored twice) gives k = 0.99712 for the re-converted
 * file: |k - 1| <= 0.01 leaves room for the basis-set (Pulay) and
 * finite-difference errors of real DFT data (0.3% at the high-pressure end,
 * where the fit is weighted) while any sign or unit error is off by >= 2.
 * The sign must agree at each of the 111 points with |tr(sigma)/3| > 0.005
 * eV/A^3 (0.8 GPa); below that, near the minimum, the SCF noise of the
 * energies dominates the slope.
 * Before the fix: the stored stress is pressure-signed, k = -0.99715 and
 * all 111 points disagree in sign (a = 2.63, the first interior point:
 * stored tr/3 = +3.27668 eV/A^3 while dE/dV = -3.2873 eV/A^3 < 0, a
 * compressed cell). */
TEST_CASE("HPO example data: eos_bcc.tadahdb stores the Cauchy stress, "
          "tr(sigma)/3 = dE/dV", "[hpo][example_data][stress_convention]")
{
    tadah::core::PeriodicTable::initialize();
    const StructureDB stdb = load_example_data("eos_bcc.tadahdb");
    REQUIRE(stdb.size() == 138);

    struct Point {
        double v, e, p;   // volume, energy, tr(sigma)/3
        std::string label;
    };
    std::vector<Point> pts;
    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        const double p =
            (st.stress(0, 0) + st.stress(1, 1) + st.stress(2, 2)) / 3.0;
        pts.push_back({cell_volume(st), st.energy, p, st.label});
    }
    std::stable_sort(pts.begin(), pts.end(),
                     [](const Point& a, const Point& b) { return a.v < b.v; });

    /* One point per volume: a structure stored twice must be the same
     * calculation (energies printed to 1e-12 eV in the file). */
    std::vector<Point> u;
    for (const Point& p : pts) {
        if (!u.empty() && std::fabs(p.v - u.back().v) <= 1e-9 * p.v) {
            INFO("'" << p.label << "' and '" << u.back().label
                     << "' have the same volume " << p.v);
            REQUIRE(std::fabs(p.e - u.back().e) <= 1e-9);
            REQUIRE(std::fabs(p.p - u.back().p) <= 1e-12);
            continue;
        }
        u.push_back(p);
    }
    REQUIRE(u.size() == 137);

    double num = 0.0, den = 0.0;
    std::size_t n_big = 0, n_disagree = 0;
    std::ostringstream disagree;
    disagree << std::setprecision(8);
    for (std::size_t i = 1; i + 1 < u.size(); ++i) {
        const double dedv = central_derivative(u[i - 1].v, u[i].v, u[i + 1].v,
                                               u[i - 1].e, u[i].e, u[i + 1].e);
        num += u[i].p * dedv;
        den += dedv * dedv;
        if (std::fabs(u[i].p) > 0.005) {
            ++n_big;
            if ((u[i].p > 0.0) != (dedv > 0.0) && n_disagree++ < 3)
                disagree << "\n  '" << u[i].label << "' V = " << u[i].v
                         << ": stored tr(sigma)/3 = " << u[i].p
                         << " eV/A^3, dE/dV = " << dedv << " eV/A^3";
        }
    }
    REQUIRE(den > 0.0);
    const double k = num / den;
    INFO(std::setprecision(8)
         << "least-squares tr(sigma)/3 = k dE/dV: k = " << k);
    INFO(n_disagree << " of " << n_big
         << " points with |tr(sigma)/3| > 0.005 eV/A^3 have the sign of "
            "-dE/dV (pressure), the first:" << disagree.str());
    REQUIRE(n_big == 111);
    CHECK(k == Approx(1.0).margin(0.01));
    CHECK(n_disagree == 0);
}

/* Pins: elastic.tadahdb stores sigma = (1/V0) dE/d(strain) for the +-1%
 * uniaxial cells along x, y and z and the +-1% shear cells xy, yz and xz,
 * the derivative a central difference through the 0 and +-2% cells of the
 * same family (the reference a0 = 3.320087 A is structure 1; families are
 * identified from the cells, not from their order). The re-converted file
 * gives FD/sigma = 0.991 (+1%) and 1.008 (-1%) uniaxial, 1.0011 shear: the
 * 3% bound covers the O(h^2) third-order term the +-1% asymmetry shows, and
 * it fails a file whose stress has the right sign but whose energies are
 * still E_wo (FD/sigma = 0.94 for the shear and the +1% uniaxial cells).
 * Before the fix: every comparison has the opposite sign, e.g. xx +1%:
 * (1/V0) dE/d(eps) = +0.014985 against stored sigma_xx = -0.015967 eV/A^3;
 * xy +1%: +0.004389 against stored sigma_xy = -0.004669 eV/A^3. */
TEST_CASE("HPO example data: elastic.tadahdb stores the Cauchy stress, "
          "(1/V0) dE/d(strain) = sigma for the uniaxial and shear cells",
          "[hpo][example_data][stress_convention]")
{
    tadah::core::PeriodicTable::initialize();
    const StructureDB stdb = load_example_data("elastic.tadahdb");
    REQUIRE(stdb.size() == 29);

    /* The reference: the one force-free cubic cell (structures 26-29, the
     * frozen phonons, have the same cell and non-zero forces). */
    std::size_t ref = stdb.size();
    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        const double a = st.cell(0, 0);
        const double cubic[3][3] = {{a, 0, 0}, {0, a, 0}, {0, 0, a}};
        if (force_free(st) && cell_is(st, cubic)) {
            REQUIRE(ref == stdb.size());   // only one
            ref = s;
        }
    }
    REQUIRE(ref < stdb.size());
    const double a0 = stdb(ref).cell(0, 0);
    const double v0 = cell_volume(stdb(ref));
    REQUIRE(a0 == Approx(3.320087).margin(1e-9));

    /* Every other force-free structure is the reference with one cell entry
     * (r, c) changed: strain eps = cell(r,r)/a0 - 1 (uniaxial) or
     * gamma = cell(r,c)/a0 (shear); its stress component is (r, c). */
    using Key = std::pair<int, int>;
    std::map<Key, std::vector<std::pair<double, std::size_t>>> families;
    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const Structure& st = stdb(s);
        if (s == ref || !force_free(st)) continue;
        INFO("structure " << s + 1 << " '" << st.label << "'");
        int n_changed = 0;
        Key key{-1, -1};
        double strain = 0.0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                const double d = st.cell(r, c) / a0 - (r == c ? 1.0 : 0.0);
                if (std::fabs(d) > 1e-6) {
                    ++n_changed;
                    key = {std::min(r, c), std::max(r, c)};
                    strain = d;
                } else {
                    REQUIRE(std::fabs(d) <= 1e-9);
                }
            }
        REQUIRE(n_changed == 1);
        families[key].push_back({strain, s});
    }
    REQUIRE(families.size() == 6);   // xx yy zz, xy yz xz

    std::size_t n_compared = 0;
    for (auto& [key, members] : families) {
        members.push_back({0.0, ref});
        std::sort(members.begin(), members.end());
        REQUIRE(members.size() == 5);   // -2% -1% 0 +1% +2%
        for (std::size_t i = 1; i + 1 < members.size(); ++i) {
            const double eps = members[i].first;
            if (std::fabs(std::fabs(eps) - 0.01) > 1e-5) continue;
            const Structure& st = stdb(members[i].second);
            const double fd = central_derivative(
                members[i - 1].first, eps, members[i + 1].first,
                stdb(members[i - 1].second).energy, st.energy,
                stdb(members[i + 1].second).energy) / v0;
            const double sigma = st.stress(key.first, key.second);
            INFO(std::setprecision(8)
                 << "structure " << members[i].second + 1 << ": strain ("
                 << key.first << "," << key.second << ") = " << eps
                 << ", (1/V0) dE/d(strain) = " << fd
                 << " eV/A^3, stored sigma = " << sigma << " eV/A^3");
            REQUIRE(std::fabs(sigma) > 1e-3);
            CHECK((fd > 0.0) == (sigma > 0.0));
            CHECK(std::fabs(fd - sigma) <= 0.03 * std::fabs(sigma));
            ++n_compared;
        }
    }
    CHECK(n_compared == 12);   // the +-1% pair of each of the six families
}

/* Pins: the stored energy is VASP's free energy F (the ionic step's "free
 * energy TOTEN") plus 3.69807734 eV/atom, at eight structures picked by
 * their cells. Expected = F (OUTCAR, 8 decimals) + 2 x 3.69807734, exact in
 * decimal; the file prints 12 decimals, so 1e-9 eV is far above any
 * rounding and far below the smallest F - E_wo here (1.16e-4 eV).
 * Before the fix: the files store E_wo + 3.69807734 eV/atom, e.g. a = 2.87:
 * -10.70456007 instead of -10.71035596 (F - E_wo = -5.8 meV/structure). */
TEST_CASE("HPO example data: eos_bcc.tadahdb and elastic.tadahdb store the "
          "free energy F + 3.69807734 eV/atom",
          "[hpo][example_data][stress_convention]")
{
    tadah::core::PeriodicTable::initialize();

    struct Expected {
        const char* file;
        const char* what;
        double rows[3][3];
        std::size_t count;     // structures with this cell (force-free)
        double energy;         // F + 2 x 3.69807734 (eV)
        double energy_before;  // E_wo + 2 x 3.69807734, the file before the fix
    };
    const double a0 = 3.320087;
    const Expected expected[] = {
        {"eos_bcc.tadahdb", "a = 2.60 (a2.60/OUTCAR)",
         {{2.60, 0, 0}, {0, 2.60, 0}, {0, 0, 2.60}}, 1,
         2.61623306, 2.61755141},
        {"eos_bcc.tadahdb", "a = 2.87 (a2.87/OUTCAR)",
         {{2.87, 0, 0}, {0, 2.87, 0}, {0, 0, 2.87}}, 1,
         -10.71035596, -10.70456007},
        {"eos_bcc.tadahdb", "a = 3.32 (a3.32/OUTCAR and MINIMA/a3.32/OUTCAR)",
         {{3.32, 0, 0}, {0, 3.32, 0}, {0, 0, 3.32}}, 2,
         -16.22789906, -16.22801539},
        {"eos_bcc.tadahdb", "a = 3.68 (a3.68/OUTCAR)",
         {{3.68, 0, 0}, {0, 3.68, 0}, {0, 0, 3.68}}, 1,
         -14.46503623, -14.46542819},
        {"elastic.tadahdb", "the reference a0 = 3.320087 (structure 1)",
         {{a0, 0, 0}, {0, a0, 0}, {0, 0, a0}}, 1,
         -16.22796184, -16.22807922},
        {"elastic.tadahdb", "xx +2% (structure 2)",
         {{3.38648874, 0, 0}, {0, a0, 0}, {0, 0, a0}}, 1,
         -16.21638161, -16.21711127},
        {"elastic.tadahdb", "xy shear +1% (structure 15)",
         {{a0, 0.03320087, 0}, {0, a0, 0}, {0, 0, a0}}, 1,
         -16.22710826, -16.22729657},
    };

    std::map<std::string, StructureDB> files;
    for (const Expected& x : expected)
        if (!files.count(x.file))
            files.emplace(x.file, load_example_data(x.file));

    std::size_t n_checked = 0;
    for (const Expected& x : expected) {
        const StructureDB& stdb = files.at(x.file);
        std::size_t n = 0;
        for (std::size_t s = 0; s < stdb.size(); ++s) {
            const Structure& st = stdb(s);
            if (!force_free(st) || !cell_is(st, x.rows)) continue;
            ++n;
            INFO(std::setprecision(12) << x.file << ", " << x.what
                 << ": structure " << s + 1 << " '" << st.label
                 << "' stores E = " << st.energy
                 << " eV; F + 2 x 3.69807734 = " << x.energy
                 << " eV (E_wo + 2 x 3.69807734 = " << x.energy_before << ")");
            CHECK(std::fabs(st.energy - x.energy) <= 1e-9);
            ++n_checked;
        }
        INFO(x.file << ", " << x.what);
        REQUIRE(n == x.count);
    }
    CHECK(n_checked == 8);
}

// test_example_data_cells.cpp
//
// Data sanity of the datasets the HPO worked examples train and validate on
// (examples/common/data, found through TADAH_HPO_EXAMPLES_DIR, as in
// test_hpo_host_e2e.cpp). Row i of Structure::cell is the lattice vector
// a_i, so a Cartesian position r has fractional coordinates
// f = (cell^T)^{-1} r. Every structure is a 2-atom bcc Ta cell read through
// StructureDB:
//   - a force-free structure (all DFT forces 0) is the perfect crystal:
//     atom 1 sits at fractional (1/2, 1/2, 1/2) from atom 0, and the eight
//     first-shell vectors of each atom sum to zero;
//   - a structure with forces (elastic.tadahdb 26-29: atom 0 moved along x)
//     is a frozen phonon: the first-shell sum of each atom, -8 times its
//     offset from the centre of its cage, points along its restoring force.
// Structures 14-25 of elastic.tadahdb (the sheared cells) were written by
// the OUTCAR reader before MLIP c6d457c, which stored the lattice vectors as
// columns: read as rows, atom 1 is off the body centre by 0.005-0.01 and
// the shell sums are 0.19-0.38 A, while their DFT forces are 0 (audit
// cell-convention 2026-09-30, core-2 / lammps-hpo-md-1).
//
// The expectations are independent of the code under test: the fractional
// coordinates come from the reciprocal vectors (a_j x a_k)/V written out
// here, and the neighbour shells from a brute-force image sum.

#include "catch2/catch.hpp"

#include <tadah/core/periodic_table.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using V3 = std::array<double, 3>;

const fs::path& example_data_dir()
{
    static const fs::path dir =
        fs::path(TADAH_HPO_EXAMPLES_DIR) / "common" / "data";
    return dir;
}

V3 sub(const V3& a, const V3& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
double dot(const V3& a, const V3& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
double norm(const V3& a) { return std::sqrt(dot(a, a)); }
V3 cross(const V3& a, const V3& b)
{
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}

V3 row(const tadah::mlip::Structure& st, int i)
{
    return {st.cell(i, 0), st.cell(i, 1), st.cell(i, 2)};
}
V3 position(const tadah::mlip::Structure& st, std::size_t a)
{
    const auto& p = st.atoms[a].position;
    return {p(0), p(1), p(2)};
}
V3 force(const tadah::mlip::Structure& st, std::size_t a)
{
    const auto& f = st.atoms[a].force;
    return {f(0), f(1), f(2)};
}

/* f_i = d . (a_j x a_k) / (a_i . (a_j x a_k)), (i, j, k) cyclic: the
 * solution of d = f_0 a_0 + f_1 a_1 + f_2 a_2 for the rows a_i. */
V3 fractional(const tadah::mlip::Structure& st, const V3& d)
{
    const V3 a[3] = {row(st, 0), row(st, 1), row(st, 2)};
    V3 f{};
    for (int i = 0; i < 3; ++i) {
        const V3 b = cross(a[(i + 1) % 3], a[(i + 2) % 3]);
        f[i] = dot(d, b) / dot(a[i], b);
    }
    return f;
}

/* Image vectors r_j + n.A - r_i from atom i to every atom j of the periodic
 * crystal with |n_k| <= 2, sorted by length (atom i itself excluded). */
std::vector<V3> neighbour_vectors(const tadah::mlip::Structure& st,
                                  std::size_t i)
{
    const V3 a[3] = {row(st, 0), row(st, 1), row(st, 2)};
    std::vector<V3> out;
    for (std::size_t j = 0; j < st.natoms(); ++j)
        for (int n0 = -2; n0 <= 2; ++n0)
            for (int n1 = -2; n1 <= 2; ++n1)
                for (int n2 = -2; n2 <= 2; ++n2) {
                    if (j == i && n0 == 0 && n1 == 0 && n2 == 0) continue;
                    V3 v = sub(position(st, j), position(st, i));
                    for (int k = 0; k < 3; ++k)
                        v[k] += n0 * a[0][k] + n1 * a[1][k] + n2 * a[2][k];
                    out.push_back(v);
                }
    std::sort(out.begin(), out.end(), [](const V3& x, const V3& y) {
        return norm(x) < norm(y);
    });
    return out;
}

struct DatasetCounts {
    std::size_t structures = 0;
    std::size_t force_free = 0;
    std::size_t with_forces = 0;
    std::size_t non_symmetric_cells = 0;
};

/* Checks every structure of `file` (see the header comment) and returns
 * what it saw, so a case can pin that the checks were not vacuous.
 *
 * Tolerances: the positions are OUTCAR's 5-decimal Cartesian coordinates,
 * so each is within 5e-6 A of the lattice site: a fractional coordinate is
 * off by at most ~2e-6 (|a| >= 2.6 A), and a sum of eight first-shell
 * vectors by at most 8 * 5e-6 * sqrt(3) = 6.9e-5 A. A transposed sheared
 * cell gives 5e-3 and 0.19 A. */
DatasetCounts check_bcc_dataset(const std::string& file)
{
    const fs::path path = example_data_dir() / file;
    INFO("dataset " << path.string());
    REQUIRE(fs::exists(path));

    tadah::mlip::StructureDB stdb;
    stdb.add(path.string());

    DatasetCounts n;
    n.structures = stdb.size();
    for (std::size_t s = 0; s < stdb.size(); ++s) {
        const tadah::mlip::Structure& st = stdb(s);
        INFO("structure " << s + 1 << " '" << st.label << "'");
        REQUIRE(st.natoms() == 2);
        CHECK(st.atoms[0].symbol == "Ta");
        CHECK(st.atoms[1].symbol == "Ta");

        double asym = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                asym = std::max(asym, std::fabs(st.cell(i, j) - st.cell(j, i)));
        if (asym > 1e-6) ++n.non_symmetric_cells;

        double fmax = 0.0;
        for (std::size_t a = 0; a < 2; ++a)
            for (double c : force(st, a)) fmax = std::max(fmax, std::fabs(c));
        const bool force_free = fmax == 0.0;
        if (force_free) ++n.force_free; else ++n.with_forces;

        if (force_free) {
            /* Atom 1 at the body centre of the cell spanned by the rows. */
            const V3 f = fractional(st, sub(position(st, 1), position(st, 0)));
            for (int k = 0; k < 3; ++k) {
                INFO("fractional coordinate " << k << " of atom 1 - atom 0: "
                     << f[k]);
                const double wrapped = f[k] - std::floor(f[k]);
                CHECK(std::fabs(wrapped - 0.5) < 1e-5);
            }
        }

        for (std::size_t i = 0; i < 2; ++i) {
            const std::vector<V3> nb = neighbour_vectors(st, i);
            REQUIRE(nb.size() > 8);
            /* A first shell of eight, well apart from the second (the
             * second bcc shell is 15% further out). */
            REQUIRE(norm(nb[8]) - norm(nb[7]) > 0.05 * norm(nb[7]));
            V3 sum{};
            for (int k = 0; k < 8; ++k)
                for (int c = 0; c < 3; ++c) sum[c] += nb[k][c];
            INFO("atom " << i << ": first-shell sum (" << sum[0] << ", "
                 << sum[1] << ", " << sum[2] << ") A");
            if (force_free) {
                CHECK(norm(sum) < 1e-4);
            } else {
                /* Frozen phonon: the sum is -8 times the atom's offset
                 * from its cage centre, and the restoring force points the
                 * same way. */
                const V3 fi = force(st, i);
                REQUIRE(norm(fi) > 0.0);
                REQUIRE(norm(sum) > 1e-3);
                CHECK(dot(sum, fi) / (norm(sum) * norm(fi)) > 0.999);
            }
        }
    }
    return n;
}

}  // namespace

/* The HPO elastic set: 29 structures, the 12 sheared ones (14-25) with
 * non-symmetric cells, four frozen phonons (26-29). Fails while structures
 * 14-25 store their cells transposed. */
TEST_CASE("HPO example data: elastic.tadahdb cells are lattice-vector rows "
          "(each force-free structure is a perfect bcc crystal)",
          "[hpo][example_data][cell_convention]")
{
    tadah::core::PeriodicTable::initialize();
    const DatasetCounts n = check_bcc_dataset("elastic.tadahdb");
    CHECK(n.structures == 29);
    CHECK(n.non_symmetric_cells == 12);
    CHECK(n.force_free == 25);
    CHECK(n.with_forces == 4);
}

/* The HPO EOS set: 138 cubic cells, all force-free. Cubic cells read the
 * same either way; this pins that the loader and the checks agree on the
 * data that is right. */
TEST_CASE("HPO example data: eos_bcc.tadahdb cells are lattice-vector rows "
          "(each structure is a perfect bcc crystal)",
          "[hpo][example_data][cell_convention]")
{
    tadah::core::PeriodicTable::initialize();
    const DatasetCounts n = check_bcc_dataset("eos_bcc.tadahdb");
    CHECK(n.structures == 138);
    CHECK(n.force_free == 138);
    CHECK(n.with_forces == 0);
}

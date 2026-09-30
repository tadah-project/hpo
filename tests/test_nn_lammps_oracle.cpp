// test_nn_lammps_oracle.cpp
//
// pair_style tadah as an oracle for the MLIP neighbour lists (workstream C,
// DESIGN-nnfinder-exact-images.md, section 5.3 (b)).
//
// LAMMPS builds its own ghost atoms and neighbour lists for any triclinic
// box (a lower-triangular cell: rows (lx,0,0), (xy,ly,0), (xz,yz,lz), the
// same rows as Structure::cell) and maps every atom into the box
// (create_atoms single ... remap yes). The MLIP prediction -- NNFinder,
// DescriptorsCalc and the model, as `tadah predict` does it -- must give the
// same total energy (1e-9 relative) and the same force on every atom
// (1e-8 eV/A, atoms matched by id) with the same potential: a small linear
// D2_BP + DM_EAD (Lmax 1) model with smooth (Cut_Cos) cutoffs at 5.5 A,
// EAMFAST false (the generic pair_style kernel, no tables).
//
// Cells (heights below the cutoff unless stated):
//   - a thick 3x3x3 bcc a = 2.9 supercell, rattled: a control that both
//     neighbour searches have always handled;
//   - 2-atom bcc a = 2.9: the body-centre neighbours at 1.66 a = 4.81 A need
//     the image -2 a0 (the old image bound was 1), so the old lists gave a
//     perfect bcc lattice a net force;
//   - a hexagonal cell a = 4, c = 3.2 with 3 atoms at general positions;
//   - the sheared representation (4,0,0),(12,4,0),(0,0,4) of a simple cubic
//     lattice, 2 atoms (LAMMPS warns about the large skew and runs);
//   - the hexagonal cell with its atoms displaced by lattice vectors: MLIP
//     reads the unwrapped positions, LAMMPS remaps them.
// Before the exact-image fix the MLIP side loses periodic images on every
// cell but the control.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/models/dc_selector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::DescriptorsCalc;
using tadah::mlip::DM_Function_Base;
using tadah::mlip::M_Tadah_Base;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::models::DC_Selector;

/* D2_BP (4 Gaussians) + DM_EAD (Lmax 1, 3 Gaussians: 6 descriptors), both
 * Cut_Cos at 5.5 A, linear, no bias: 10 weights. */
const char* kPotential =
    "ATOM              Ta\n"
    "WATOM             1\n"
    "BIAS              false\n"
    "NORM              false\n"
    "DIMER             false 0 false\n"
    "EAMFAST           false\n"
    "INIT2B            true\n"
    "TYPE2B            D2_BP 4 4 Ta Ta\n"
    "CGRID2B           2.2 3.2 4.2 5.0\n"
    "SGRID2B           1.5 1.0 1.0 1.5\n"
    "RCTYPE2B          Cut_Cos\n"
    "RCUT2B            5.5\n"
    "INITMB            true\n"
    "TYPEMB            DM_EAD 1 3 3 Ta Ta\n"
    "CGRIDMB           2.5 3.5 4.5\n"
    "SGRIDMB           1.0 1.0 1.0\n"
    "RCTYPEMB          Cut_Cos\n"
    "RCUTMB            5.5\n"
    "MODEL             M_BLR BF_Linear\n"
    "LAMBDA            0\n"
    "WEIGHTS           6.5 -3.5 1.25 -0.4 -0.3 0.175 -0.06 0.25 -0.15 0.075\n";

Context parse_potential(bool forces) {
    std::istringstream in(kPotential);
    Context ctx = tadah::core::CoreUserConfigParser().parse(in);
    ctx.remove("FORCE");
    ctx.add("FORCE", forces ? "true" : "false");
    ctx.remove("STRESS");
    ctx.add("STRESS", "false");
    return ctx;
}

/* The potential as `tadah predict -F` evaluates it: NNFinder at RCUTMAX,
 * DescriptorsCalc and the model. */
class MlipPotential {
public:
    MlipPotential()
        : ctx_(parse_potential(true)), dcs_(ctx_),
          dc_(ctx_, *dcs_.d2b, *dcs_.dmb, *dcs_.c2b, *dcs_.cmb),
          fb_(tadah::core::factory<DM_Function_Base, const Context&>(
              ctx_.get<std::string>("MODEL", 1), ctx_)),
          model_(tadah::core::factory<M_Tadah_Base, DM_Function_Base&, Context&>(
              ctx_.get<std::string>("MODEL", 0), *fb_, ctx_)) {}

    /* Predicted total energy and forces of `st`. */
    Structure predict(Structure st) {
        NNFinder().calc(st, ctx_);
        return model_->predict(ctx_, st, dc_);
    }

private:
    Context ctx_;
    DC_Selector dcs_;
    DescriptorsCalc<> dc_;
    std::unique_ptr<DM_Function_Base> fb_;
    std::unique_ptr<M_Tadah_Base> model_;
};

using Rows = std::array<std::array<double, 3>, 3>;
using Frac = std::array<double, 3>;

/* A cell with lower-triangular rows (LAMMPS' restricted triclinic box) and
 * atoms at fractional positions plus integer lattice offsets. */
struct Cell {
    std::string name;
    Rows rows;
    std::vector<Frac> fracs;
    std::vector<std::array<int, 3>> offsets;   // empty: none
};

Structure to_structure(const Cell& c) {
    Structure st;
    const Rows& a = c.rows;
    st.cell.load(a[0][0], a[0][1], a[0][2],
                 a[1][0], a[1][1], a[1][2],
                 a[2][0], a[2][1], a[2][2]);
    const tadah::core::Element ta = PeriodicTable::find_by_symbol("Ta");
    for (std::size_t i = 0; i < c.fracs.size(); ++i) {
        Frac f = c.fracs[i];
        if (!c.offsets.empty())
            for (int k = 0; k < 3; ++k) f[k] += c.offsets[i][k];
        double r[3];
        for (int d = 0; d < 3; ++d)
            r[d] = f[0] * a[0][d] + f[1] * a[1][d] + f[2] * a[2][d];
        st.add_atom(Atom(ta, r[0], r[1], r[2]));
    }
    return st;
}

std::string num(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

/* The LAMMPS script of @p st: a prism box from its rows, one
 * create_atoms single per atom (ids 1..N in order, remapped into the box),
 * pair_style tadah with the staged potential, run 0. */
std::vector<std::string> lammps_script(const Structure& st) {
    std::vector<std::string> s = {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "region box prism 0 " + num(st.cell(0, 0)) + " 0 " + num(st.cell(1, 1)) +
            " 0 " + num(st.cell(2, 2)) + " " + num(st.cell(1, 0)) + " " +
            num(st.cell(2, 0)) + " " + num(st.cell(2, 1)) + " units box",
        "create_box 1 box",
    };
    for (const Atom& at : st.atoms)
        s.push_back("create_atoms 1 single " + num(at.position[0]) + " " +
                    num(at.position[1]) + " " + num(at.position[2]) +
                    " units box remap yes");
    s.push_back("mass 1 180.94788");
    s.push_back("pair_style tadah");
    s.push_back("pair_coeff * * EXTERN Ta");
    s.push_back("neighbor 1.0 nsq");
    s.push_back("variable e equal pe");
    s.push_back("variable n equal atoms");
    for (std::size_t i = 1; i <= st.natoms(); ++i) {
        const std::string k = std::to_string(i);
        for (const char* c : {"fx", "fy", "fz"})
            s.push_back(std::string("variable ") + c + k + " equal " + c + "[" +
                        k + "]");
    }
    s.push_back("run 0 post no");
    return s;
}

std::vector<Cell> make_cells() {
    const double s3 = std::sqrt(3.0);
    const Rows bcc = {{{2.9, 0.0, 0.0}, {0.0, 2.9, 0.0}, {0.0, 0.0, 2.9}}};
    const Rows hex = {{{4.0, 0.0, 0.0}, {2.0, 2.0 * s3, 0.0}, {0.0, 0.0, 3.2}}};
    const Rows shear = {{{4.0, 0.0, 0.0}, {12.0, 4.0, 0.0}, {0.0, 0.0, 4.0}}};
    const std::vector<Frac> hex_fracs = {
        {0.02, 0.05, 0.10}, {0.36, 0.68, 0.55}, {0.71, 0.29, 0.83}};

    std::vector<Cell> cells;

    /* Control: 3x3x3 bcc a = 2.9 (heights 8.7 >= 5.5), atoms inside the
     * cell and rattled deterministically. */
    {
        Cell c{"thick 3x3x3 bcc a=2.9, rattled (control)",
               {{{8.7, 0.0, 0.0}, {0.0, 8.7, 0.0}, {0.0, 0.0, 8.7}}}, {}, {}};
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y)
                for (int z = 0; z < 3; ++z)
                    for (double h : {0.0, 0.5}) {
                        const double i = static_cast<double>(c.fracs.size());
                        const int n[3] = {x, y, z};
                        Frac f{};
                        for (int k = 0; k < 3; ++k)
                            f[k] = (n[k] + h) / 3.0 + 0.01 +
                                   0.002 * std::sin(1.9 * i + 0.8 * k);
                        c.fracs.push_back(f);
                    }
        cells.push_back(c);
    }
    cells.push_back({"bcc a=2.9, 2 atoms", bcc,
                     {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}}, {}});
    cells.push_back({"hexagonal a=4 c=3.2, 3 atoms", hex, hex_fracs, {}});
    cells.push_back({"sheared (4,0,0),(12,4,0),(0,0,4), 2 atoms", shear,
                     {{0.1, 0.2, 0.3}, {0.6, 0.45, 0.8}}, {}});
    cells.push_back({"hexagonal a=4 c=3.2 with lattice offsets (unwrapped)", hex,
                     hex_fracs, {{2, -3, 1}, {-1, 4, -2}, {5, 0, -3}}});
    return cells;
}

}  // namespace

TEST_CASE("pair_style tadah and the MLIP prediction agree on thin triclinic "
          "and unwrapped cells", "[nn][lammps_oracle]")
{
    PeriodicTable::initialize();
    const std::vector<Cell> cells = make_cells();

    for (const Cell& cell : cells) {
        DYNAMIC_SECTION(cell.name) {
            const Structure st = to_structure(cell);
            REQUIRE(st.natoms() == cell.fracs.size());

            /* MLIP */
            MlipPotential pot;
            const Structure pred = pot.predict(st);
            REQUIRE(pred.atoms.size() == st.natoms());

            /* LAMMPS */
            LammpsRunner runner;
            auto lmp = runner.create_instance(
                0, {"lmp", "-log", "none", "-screen", "none"});
            REQUIRE(lmp != nullptr);
            REQUIRE(runner.run_script(lmp, lammps_script(st),
                                      parse_potential(false)));
            REQUIRE(std::stoi(runner.get_variable(lmp, "n")) ==
                    static_cast<int>(st.natoms()));
            const double e_lmp = std::stod(runner.get_variable(lmp, "e"));

            INFO("MLIP energy " << num(pred.energy) << ", LAMMPS energy "
                                << num(e_lmp));
            REQUIRE(std::fabs(e_lmp) > 1.0);   // a relative comparison
            CHECK(pred.energy == Approx(e_lmp).epsilon(1e-9));

            double max_df = 0.0;
            for (std::size_t i = 0; i < st.natoms(); ++i) {
                const std::string id = std::to_string(i + 1);
                const double f_lmp[3] = {
                    std::stod(runner.get_variable(lmp, "fx" + id)),
                    std::stod(runner.get_variable(lmp, "fy" + id)),
                    std::stod(runner.get_variable(lmp, "fz" + id))};
                for (int k = 0; k < 3; ++k) {
                    CAPTURE(i, k, pred.atoms[i].force[k], f_lmp[k]);
                    const double df = std::fabs(pred.atoms[i].force[k] - f_lmp[k]);
                    max_df = std::max(max_df, df);
                    CHECK(df <= 1e-8);
                }
            }
            INFO("largest force difference " << max_df);
            CHECK(max_df <= 1e-8);
        }
    }
}

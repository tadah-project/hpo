// test_pair_tadah_nonlinear.cpp
//
// pair_style tadah with a trained non-linear model: M_KRR with Kern_RBF,
// Kern_Polynomial, Kern_Quadratic, Kern_LQ and Kern_Sigmoid (their SBASIS
// basis vectors in BASIS), and M_BLR with BF_Polynomial2. Each model is
// trained in process (mlip::Trainer, as `tadah train` and `tadah hpo`
// train), its potential (get_param_file) is staged for `pair_coeff * *
// EXTERN Ta`, and `run 0` on a held-out triclinic cell must give the
// energy, forces and stress the trained model predicts for that cell
// (M_Tadah_Base::predict). pair_style tadah evaluates non-linear models in
// compute_2b_mb_full; the comparison covers the potential's round trip
// through the context, the BASIS reading and the kernel derivatives in
// the pair style. Kern_RBF is also checked with NORM true (the pair style
// normalises the descriptors with NMEAN/NSTDEV before the kernel).
// The same comparison runs with BIAS true, each model in two LAMMPS
// instances of one process, with a linear many-body model added
// (compute_2b_mb_half sets the bias component as compute_2b_mb_full does).
//
// Before the fix: training threw ("ColumnLayout: total_cols=3 disagrees
// with design-matrix width=5"; 6 for BF_Polynomial2), so there was no
// potential to compare.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <atom.h>
#include <force.h>
#include <input.h>
#include <pair.h>
#include <pair_tadah.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::Element;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
using LMPPtr = LammpsRunner::LMPPtr;

constexpr double kRcut = 5.0;

struct ModelSpec {
    std::string name;
    std::vector<std::string> model;
    std::vector<std::string> mparams;
    bool kernel;
    bool norm;
    bool mb = false;  // DM_EAD many-body descriptor (INITMB true)
};

/* The models and MPARAMS of MLIP's test_nonlinear_model_training.cpp. */
const std::vector<ModelSpec> kModels = {
    {"M_KRR Kern_RBF", {"M_KRR", "Kern_RBF", "1"}, {"1.0"}, true, false},
    {"M_KRR Kern_RBF, NORM true", {"M_KRR", "Kern_RBF", "1"}, {"1.0"}, true,
     true},
    {"M_KRR Kern_Polynomial", {"M_KRR", "Kern_Polynomial", "1"},
     {"3", "0.5", "1.0"}, true, false},
    {"M_KRR Kern_Quadratic", {"M_KRR", "Kern_Quadratic", "1"}, {}, true,
     false},
    {"M_KRR Kern_LQ", {"M_KRR", "Kern_LQ", "1"}, {}, true, false},
    {"M_KRR Kern_Sigmoid", {"M_KRR", "Kern_Sigmoid", "1"}, {"0.5", "-1.0"},
     true, false},
    {"M_BLR BF_Polynomial2", {"M_BLR", "BF_Polynomial2"}, {}, false, false},
};

/* The models of the BIAS true case: those above, which pair_style tadah
 * evaluates in compute_2b_mb_full, and a linear many-body model, M_BLR
 * BF_Linear on D2_Blip and DM_EAD, which it evaluates in
 * compute_2b_mb_half. */
std::vector<ModelSpec> bias_models() {
    std::vector<ModelSpec> v = kModels;
    v.push_back({"M_BLR BF_Linear, DM_EAD", {"M_BLR", "BF_Linear"}, {}, false,
                 false, true});
    return v;
}

/* D2_Blip with 3 functions on Ta, Cut_Cos 5 A, WATOM 1, FORCE and STRESS
 * true, LAMBDA 0, OALGO 1, SBASIS 5 for a kernel, BIAS @p bias; with m.mb
 * also DM_EAD (Lmax 1, 2 functions, Cut_Cos 5 A). */
Context make_context(const ModelSpec& m, bool bias = false) {
    Context c;
    c.add("ATOM", "Ta");
    c.add("WATOM", 1);
    c.add("INIT2B", "true");
    c.add("INITMB", m.mb ? "true" : "false");
    c.add("TYPE2B", "D2_Blip");
    c.add("TYPE2B", 3);
    c.add("TYPE2B", 3);
    c.add("TYPE2B", "Ta");
    c.add("TYPE2B", "Ta");
    c.add("CGRID2B", "2.8 3.5 4.2");
    c.add("SGRID2B", "2.0 2.0 2.0");
    if (m.mb) {
        c.add("TYPEMB", "DM_EAD");
        c.add("TYPEMB", 1);
        c.add("TYPEMB", 2);
        c.add("TYPEMB", 2);
        c.add("TYPEMB", "Ta");
        c.add("TYPEMB", "Ta");
        c.add("CGRIDMB", "2.5 3.5");
        c.add("SGRIDMB", "1.2 0.9");
    } else {
        c.add("TYPEMB", "DM_Dummy");
        c.add("TYPEMB", "Ta");
        c.add("TYPEMB", "Ta");
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B", kRcut);
    c.add("RCUTMB", kRcut);
    c.add("DIMER", "false");
    c.add("DIMER", 0);
    c.add("DIMER", "false");
    c.add("FORCE", "true");
    c.add("STRESS", "true");
    c.add("ESIGMA", 1.0);
    c.add("FSIGMA", 0.1);
    c.add("FSIGMA", 0.0);
    c.add("SSIGMA", 0.5);
    c.add("SSIGMA", 0.0);
    c.add("BIAS", bias ? "true" : "false");
    c.add("NORM", m.norm ? "true" : "false");
    c.add("VERBOSE", 0);
    for (const auto& t : m.model) c.add("MODEL", t);
    for (const auto& t : m.mparams) c.add("MPARAMS", t);
    c.add("LAMBDA", 0);
    c.add("OALGO", 1);
    if (m.kernel) c.add("SBASIS", 5);
    return c;
}

/* A cell with rows a*[(1,0,0), (s0,1,0), (s1,s2,1)] (LAMMPS's triclinic
 * convention: xy = s0 a, xz = s1 a, yz = s2 a) and atoms at the given
 * fractional coordinates. */
Structure make_cell(double a, std::array<double, 3> shear,
                    const std::vector<std::array<double, 3>>& frac) {
    Structure st;
    st.cell.load(a, 0.0, 0.0, shear[0] * a, a, 0.0, shear[1] * a,
                 shear[2] * a, a);
    for (const auto& f : frac) {
        double r[3];
        for (int k = 0; k < 3; ++k)
            r[k] = f[0] * st.cell(0, k) + f[1] * st.cell(1, k) +
                   f[2] * st.cell(2, k);
        st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0, 0, 0));
    }
    return st;
}

/* Six sheared, rattled cells of 3 and 4 atoms with arbitrary smooth
 * targets (as in MLIP's test). */
StructureDB training_db() {
    std::vector<Structure> cells = {
        make_cell(3.30, {0.05, 0.02, -0.03},
                  {{0.00, 0.01, 0.02}, {0.48, 0.52, 0.05}, {0.27, 0.55, 0.51}}),
        make_cell(3.45, {-0.04, 0.03, 0.06},
                  {{0.01, -0.02, 0.00}, {0.52, 0.47, 0.03},
                   {0.49, 0.02, 0.53}, {-0.03, 0.51, 0.48}}),
        make_cell(3.15, {0.07, -0.05, 0.02},
                  {{0.02, -0.01, 0.00}, {0.33, 0.31, 0.36}, {0.68, 0.64, 0.70}}),
        make_cell(3.60, {0.00, 0.08, -0.02},
                  {{0.00, 0.00, 0.03}, {0.46, 0.50, 0.01},
                   {0.53, 0.04, 0.47}, {0.02, 0.55, 0.52}}),
        make_cell(3.22, {-0.06, 0.01, 0.04},
                  {{0.01, 0.03, -0.02}, {0.51, 0.43, 0.10}, {0.22, 0.61, 0.57}}),
        make_cell(3.52, {0.03, -0.02, -0.05},
                  {{-0.02, 0.01, 0.01}, {0.49, 0.53, -0.02},
                   {0.47, -0.03, 0.50}, {0.04, 0.48, 0.55}}),
    };
    StructureDB db;
    for (std::size_t s = 0; s < cells.size(); ++s) {
        Structure& st = cells[s];
        const double ds = static_cast<double>(s);
        st.energy = -8.1 * static_cast<double>(st.natoms()) +
                    0.37 * std::sin(1.3 * ds + 0.2);
        for (std::size_t a = 0; a < st.natoms(); ++a)
            for (int k = 0; k < 3; ++k)
                st.atoms[a].force[k] = 0.25 * std::sin(
                    0.7 * static_cast<double>(a) + 1.1 * k + 0.3 * ds);
        for (int x = 0; x < 3; ++x)
            for (int y = x; y < 3; ++y) {
                const double v = 0.02 * std::cos(0.9 * x + 1.7 * y + 0.5 * ds);
                st.stress(x, y) = v;
                st.stress(y, x) = v;
            }
        db.add(st);
    }
    return db;
}

/* The held-out cell (3 atoms; every coordinate inside [0, a) after the
 * shear, so LAMMPS creates the atoms where they are). */
Structure held_out_cell() {
    return make_cell(3.38, {0.04, -0.03, 0.05},
                     {{0.03, 0.02, 0.01}, {0.45, 0.53, 0.08},
                      {0.31, 0.58, 0.49}});
}

std::string num(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

/* The held-out cell as a LAMMPS script up to `run 0`. */
std::vector<std::string> cell_script(const Structure& st) {
    std::vector<std::string> s = {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "region box prism 0 " + num(st.cell(0, 0)) + " 0 " +
            num(st.cell(1, 1)) + " 0 " + num(st.cell(2, 2)) + " " +
            num(st.cell(1, 0)) + " " + num(st.cell(2, 0)) + " " +
            num(st.cell(2, 1)) + " units box",
        "create_box 1 box",
    };
    for (std::size_t a = 0; a < st.natoms(); ++a)
        s.push_back("create_atoms 1 single " + num(st.atoms[a].position[0]) +
                    " " + num(st.atoms[a].position[1]) + " " +
                    num(st.atoms[a].position[2]) + " units box remap yes");
    s.push_back("mass 1 180.94788");
    s.push_back("pair_style tadah");
    s.push_back("pair_coeff * * EXTERN Ta");
    s.push_back("neighbor 1.0 nsq");
    s.push_back("thermo_style custom step pe press");
    return s;
}

/* Stages @p pot for `pair_coeff * * EXTERN` while it lives. */
struct StagedPotential {
    explicit StagedPotential(const Context& pot) {
        LAMMPS_NS::PairTadah::stage_extern(
            std::make_shared<tadah::LammpsTadah>(pot));
    }
    ~StagedPotential() { LAMMPS_NS::PairTadah::clear_staged_extern(); }
    StagedPotential(const StagedPotential&) = delete;
    StagedPotential& operator=(const StagedPotential&) = delete;
};

struct LammpsResult {
    std::string error;
    bool linear = true;
    bool initmb = false;
    double e = 0;
    std::vector<double> f;            // 3 per atom, by tag
    std::array<double, 6> virial{};   // xx yy zz xy xz yz
    std::size_t nall = 0;             // nlocal + nghost
    std::size_t aeds_cols = 0;        // columns of LammpsTadah::aeds
    std::size_t nonzero_past_atoms = 0;  // aeds entries != 0 in columns >= nall
};

LammpsResult run0(const Structure& st, const Context& pot) {
    LammpsResult out;
    auto lmp = LammpsRunner().create_instance(
        {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    StagedPotential staged(pot);
    std::vector<std::string> lines = cell_script(st);
    lines.push_back("run 0 post no");
    for (const std::string& line : lines) {
        try {
            lmp->input->one(line);
        } catch (const std::exception& e) {
            out.error = "'" + line + "': " + e.what();
            return out;
        }
    }
    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    const tadah::LammpsTadah* lt = pt->lammps_tadah();
    out.linear = lt->linear;
    out.initmb = lt->initmb;
    out.e = lmp->force->pair->eng_vdwl;
    const int nlocal = lmp->atom->nlocal;
    REQUIRE(static_cast<std::size_t>(nlocal) == st.natoms());
    out.nall = static_cast<std::size_t>(nlocal + lmp->atom->nghost);
    out.aeds_cols = lt->aeds.cols();
    for (std::size_t n = out.nall; n < lt->aeds.cols(); ++n)
        for (std::size_t k = 0; k < lt->aeds.rows(); ++k)
            if (lt->aeds(k, n) != 0.0) ++out.nonzero_past_atoms;
    out.f.assign(3 * st.natoms(), 0.0);
    for (int i = 0; i < nlocal; ++i) {
        const auto t = static_cast<std::size_t>(lmp->atom->tag[i] - 1);
        for (int k = 0; k < 3; ++k) out.f[3 * t + k] = lmp->atom->f[i][k];
    }
    for (int k = 0; k < 6; ++k) out.virial[k] = lmp->force->pair->virial[k];
    return out;
}

/* `run 0` (l) against the model's prediction (p) for the cell st: energy
 * to 1e-10 relative, forces to 1e-8 eV/A, -virial/V against the predicted
 * Cauchy stress to 1e-9 eV/A^3, and max |F| > fmin eV/A (the forces are
 * not all small next to their tolerance). */
void check_against_prediction(const LammpsResult& l, const Structure& p,
                              const Structure& st, double fmin) {
    INFO("energy: LAMMPS " << num(l.e) << ", model " << num(p.energy));
    CHECK(l.e == Approx(p.energy).epsilon(1e-10));
    double fmax = 0;
    for (std::size_t a = 0; a < st.natoms(); ++a)
        for (int k = 0; k < 3; ++k) {
            INFO("atom " << a << " axis " << k);
            CHECK(l.f[3 * a + k] == Approx(p.atoms[a].force[k]).margin(1e-8));
            fmax = std::max(fmax, std::fabs(p.atoms[a].force[k]));
        }
    CHECK(fmax > fmin);
    const double V = st.get_volume();
    const int xx[6] = {0, 1, 2, 0, 0, 1};
    const int yy[6] = {0, 1, 2, 1, 2, 2};
    for (int k = 0; k < 6; ++k) {
        INFO("stress " << xx[k] << yy[k] << ": LAMMPS -virial/V "
                       << -l.virial[k] / V << ", model "
                       << p.stress(xx[k], yy[k]));
        CHECK(-l.virial[k] / V == Approx(p.stress(xx[k], yy[k])).margin(1e-9));
    }
}

}  // namespace

/* Pins: for each trained non-linear model (and Kern_RBF with NORM true),
 * `run 0` of pair_style tadah on the held-out 3-atom triclinic cell gives
 * the model's predicted energy (1e-10 relative), forces (1e-8 eV/A) and
 * stress, -virial/V against the predicted Cauchy stress (1e-9 eV/A^3); the
 * pair style takes its non-linear path (LammpsTadah::linear false); the
 * forces are not all small (max |F| > 0.5 eV/A).
 * Before the fix: training threw (the ColumnLayout check), for every
 * model. */
TEST_CASE("pair_style tadah evaluates a trained non-linear model as the "
          "model predicts",
          "[pair_tadah][nonlinear]") {
    PeriodicTable::initialize();
    for (const ModelSpec& m : kModels) {
        DYNAMIC_SECTION(m.name) {
            MLIPWorkspaceManager ws;
            Context c = make_context(m);
            Trainer tr(c, ws);
            StructureDB db = training_db();
            REQUIRE_NOTHROW(tr.train(db));
            const Context pot = tr.model->get_param_file();

            Structure st = held_out_cell();
            NNFinder().calc(st, kRcut);
            Context pc = tr.context;
            const Structure p = tr.model->predict(pc, st, tr.dc);

            const LammpsResult l = run0(st, pot);
            INFO("LAMMPS: " << l.error);
            REQUIRE(l.error.empty());
            CHECK_FALSE(l.linear);
            check_against_prediction(l, p, st, 0.5);
        }
    }
}

/* Pins: with BIAS true (ATOM Ta: descriptor component 0 is 1 for every
 * atom), for each trained non-linear model (as above, compute_2b_mb_full:
 * LammpsTadah::linear false) and a linear many-body model, M_BLR BF_Linear
 * on D2_Blip and DM_EAD (compute_2b_mb_half: linear and initmb true),
 * `run 0` on the held-out cell gives the model's predicted energy, forces
 * and stress (the tolerances above, but max |F| > 0.02 eV/A: with the
 * bias component most fits have smaller forces, max |F| down to 0.045
 * eV/A when the test was written) in each of two LAMMPS instances
 * created one after the other; the sections run one after another in one
 * process, so every instance but the first of the case follows another;
 * and after each run every entry of LammpsTadah::aeds in the columns past
 * the atoms LAMMPS has (nlocal + nghost; aeds has atom->nmax columns) is
 * 0: the bias component is set for those atoms only.
 * Before the fix: the bias loops of compute_2b_mb_full and
 * compute_2b_mb_half ran over every column of aeds, reading atom->type
 * past the atoms: in the first instance (M_KRR Kern_RBF) the 16095
 * columns past the 289 atoms each held a 1 (nonzero_past_atoms 16095),
 * and the second `run 0` died with SIGSEGV in compute_2b_mb_full
 * (atom->type is not set past the atoms), which ends the run; the BF_Linear
 * section run alone (-c) failed the same way in compute_2b_mb_half. */
TEST_CASE("pair_style tadah with BIAS true evaluates a trained model as the "
          "model predicts in every LAMMPS instance of a process",
          "[pair_tadah][nonlinear][bias]") {
    PeriodicTable::initialize();
    for (const ModelSpec& m : bias_models()) {
        DYNAMIC_SECTION(m.name) {
            MLIPWorkspaceManager ws;
            Context c = make_context(m, true);
            Trainer tr(c, ws);
            StructureDB db = training_db();
            REQUIRE_NOTHROW(tr.train(db));
            const Context pot = tr.model->get_param_file();

            Structure st = held_out_cell();
            NNFinder().calc(st, kRcut);
            Context pc = tr.context;
            const Structure p = tr.model->predict(pc, st, tr.dc);

            for (int instance = 1; instance <= 2; ++instance) {
                INFO("LAMMPS instance " << instance << " of the section");
                const LammpsResult l = run0(st, pot);
                INFO("LAMMPS: " << l.error);
                REQUIRE(l.error.empty());
                CHECK(l.linear == (m.model[1] == "BF_Linear"));
                CHECK(l.initmb == m.mb);
                check_against_prediction(l, p, st, 0.02);
                INFO("aeds columns " << l.aeds_cols << ", atoms (nlocal + "
                     "nghost) " << l.nall);
                REQUIRE(l.aeds_cols > l.nall);
                CHECK(l.nonzero_past_atoms == 0);
            }
        }
    }
}

// test_multispecies_bias.cpp
//
// BIAS true with two species (Nb, Ta): one one-hot intercept column per ATOM
// species on every path -- tadah train/predict (MLIP), pair_style tadah
// (Tadah.LAMMPS) and tadah hpo.
//
// The training path gave every atom one intercept column shared by the
// species (DSIZE = 1 + descriptors) while pair_style tadah builds one per
// species, so every multi-species BIAS potential gave LAMMPS energies about
// 80 times too large; HPO's analytical gradient read N + D columns of each
// atom's 1 + D descriptor.
//
// The LAMMPS cases compare pair_style tadah with the MLIP prediction (as
// `tadah predict -F` makes it) on 2x2x2 bcc Nb-Ta cells (a = 3.3 A,
// rattled): the total energy to 1e-9 relative and the force on every atom
// to 1e-8 eV/A (atoms matched by id), with LAMMPS type 1 = Ta and type 2 =
// Nb (`pair_coeff * * EXTERN Ta Nb`), the reverse of ATOM Nb Ta. One case
// per kernel of PairTadah::compute(): compute_2b (linear, two-body),
// compute_2b_mb_half (linear, many-body), compute_2b_mb_full (BF_Polynomial2)
// and compute_dimers (DIMER).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/dloss_dtheta.h>
#include <tadah/hpo/hpo_host.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/startup_validation.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/descriptors_calc.h>
#include <tadah/mlip/design_matrix/functions/dm_function_base.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/models/m_tadah_base.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>
#include <tadah/models/dc_selector.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <dlib/matrix.h>
#include <atom.h>
#include <force.h>
#include <input.h>
#include <pair_tadah.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::DescriptorsCalc;
using tadah::mlip::DM_Function_Base;
using tadah::mlip::M_Tadah_Base;
using tadah::mlip::NNFinder;
using tadah::mlip::StDescriptors;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;
using tadah::models::DC_Selector;
using LMPPtr = LammpsRunner::LMPPtr;

constexpr double kRcut = 5.0;
constexpr int kNb = 41;
constexpr int kTa = 73;

std::string num(double v) {
  std::ostringstream o;
  o << std::setprecision(17) << v;
  return o.str();
}

/* ATOM Nb Ta, BIAS true, D2_BP (3 Gaussians, every pair; with `mb` also
 * DM_EAD, Lmax 1, 2 Gaussians: 4 descriptors), Cut_Cos at 5 A, M_BLR
 * BF_Linear, LAMBDA 0, OALGO 1, EAMFAST false, energies only. */
std::vector<std::string> nbta_lines(bool mb, bool bias = true) {
  std::vector<std::string> l = {
      "ATOM Nb Ta", std::string("BIAS ") + (bias ? "true" : "false"),
      "NORM false", "DIMER false 0 false", "EAMFAST false", "INIT2B true",
      "TYPE2B D2_BP 3 3 * *", "CGRID2B 2.4 3.1 3.9", "SGRID2B 1.0 1.2 0.9",
      "RCTYPE2B Cut_Cos", "RCUT2B 5.0", "FORCE false", "STRESS false",
      "LAMBDA 0", "OALGO 1",
      "VERBOSE 0", "MODEL M_BLR BF_Linear"};
  if (mb) {
    for (const char *s : {"INITMB true", "TYPEMB DM_EAD 1 2 2 * *",
                          "CGRIDMB 2.6 3.6", "SGRIDMB 1.0 1.1",
                          "RCTYPEMB Cut_Cos", "RCUTMB 5.0"})
      l.push_back(s);
  } else {
    l.push_back("INITMB false");
  }
  return l;
}

Context parse(const std::vector<std::string> &lines) {
  return tadah::core::CoreUserConfigParser().parse(lines);
}

void set(Context &c, const std::string &key, const std::string &value) {
  c.remove(key);
  c.add(key, value);
}

/* The potential as a file, read back (no internal keys). */
Context reload(const Context &pot) {
  std::ostringstream os;
  pot.write_potential(os, 18);
  std::istringstream in(os.str());
  return tadah::core::CoreUserConfigParser().parse(in);
}

/* A 2x2x2 bcc cell (a = 3.3 A) with the given element on each of its 16
 * sites, every atom moved deterministically by up to 0.1 A. */
Structure bcc_cell(const std::vector<std::string> &sites, unsigned seed) {
  Structure st;
  const double L = 6.6;
  st.cell.load(L, 0.0, 0.0, 0.0, L, 0.0, 0.0, 0.0, L);
  std::size_t i = 0;
  for (int x = 0; x < 2; ++x)
    for (int y = 0; y < 2; ++y)
      for (int z = 0; z < 2; ++z)
        for (double h : {0.0, 0.5}) {
          const double n[3] = {double(x), double(y), double(z)};
          double r[3];
          for (int k = 0; k < 3; ++k)
            r[k] = 3.3 * (n[k] + h) +
                   0.1 * std::sin(1.3 * double(i) + 0.7 * k + 0.37 * seed);
          st.add_atom(Atom(PeriodicTable::find_by_symbol(sites[i]), r[0],
                           r[1], r[2], 0.0, 0.0, 0.0));
          ++i;
        }
  return st;
}

/* n_nb Nb atoms on sites drawn by a Fisher-Yates shuffle of
 * std::mt19937(seed), Ta on the others. */
Structure tanb_cell(int n_nb, unsigned seed) {
  std::vector<std::string> sites(16, "Ta");
  std::vector<int> order(16);
  for (int k = 0; k < 16; ++k) order[k] = k;
  std::mt19937 rng(seed);
  for (int k = 15; k > 0; --k) {
    const int j = static_cast<int>(rng() % static_cast<std::uint32_t>(k + 1));
    std::swap(order[k], order[j]);
  }
  for (int k = 0; k < n_nb; ++k) sites[order[k]] = "Nb";
  return bcc_cell(sites, seed);
}

int count_of(const Structure &st, int Z) {
  int n = 0;
  for (const Atom &a : st.atoms) n += a.Z == Z;
  return n;
}

/* Energies: -7.6 n_Nb - 8.9 n_Ta + w . (sum of the BIAS false descriptors)
 * + 0.01 sin(1.7 s + 0.3), on cells with the given Nb counts. */
StructureDB make_db(const std::vector<int> &n_nb, bool mb, unsigned seed0) {
  MLIPWorkspaceManager ws;
  Context c0 = parse(nbta_lines(mb, false));
  Trainer truth(c0, ws);
  const std::vector<double> w = {0.11, -0.07, 0.035, 0.02, -0.013, 0.008,
                                 -0.004};
  StructureDB db;
  for (std::size_t s = 0; s < n_nb.size(); ++s) {
    Structure st = tanb_cell(n_nb[s], seed0 + static_cast<unsigned>(s));
    NNFinder().calc(st, kRcut);
    const StDescriptors d = truth.dc.calc(st);
    double e = -7.6 * count_of(st, kNb) - 8.9 * count_of(st, kTa) +
               0.01 * std::sin(1.7 * double(s) + 0.3);
    for (std::size_t a = 0; a < d.naed(); ++a)
      for (std::size_t k = 0; k < d.dim(); ++k) e += w[k] * d.get_aed(a)(k);
    st.energy = e;
    db.add(st);
  }
  db.dbidx = {0, db.size()};
  return db;
}

/* The potential as `tadah predict -F` evaluates it. */
struct MlipPotential {
  Context ctx;
  DC_Selector dcs;
  DescriptorsCalc<> dc;
  std::unique_ptr<DM_Function_Base> fb;
  std::unique_ptr<M_Tadah_Base> model;
  MlipPotential(const Context &pot, bool forces)
      : ctx(with_forces(pot, forces)), dcs(ctx),
        dc(ctx, *dcs.d2b, *dcs.dmb, *dcs.c2b, *dcs.cmb),
        fb(tadah::core::factory<DM_Function_Base, const Context &>(
            ctx.get<std::string>("MODEL", 1), ctx)),
        model(tadah::core::factory<M_Tadah_Base, DM_Function_Base &,
                                   Context &>(
            ctx.get<std::string>("MODEL", 0), *fb, ctx)) {}
  static Context with_forces(Context c, bool forces) {
    set(c, "FORCE", forces ? "true" : "false");
    set(c, "STRESS", "false");
    return c;
  }
  Structure predict(Structure st) {
    NNFinder().calc(st, ctx);
    return model->predict(ctx, st, dc);
  }
};

LMPPtr make_instance() {
  return LammpsRunner().create_instance(
      {"lmp", "-log", "none", "-screen", "none"});
}

/* Stages @p pot for `pair_coeff * * EXTERN` while it lives. */
struct StagedPotential {
  explicit StagedPotential(const Context &pot) {
    LAMMPS_NS::PairTadah::stage_extern(
        std::make_shared<tadah::LammpsTadah>(pot));
  }
  ~StagedPotential() { LAMMPS_NS::PairTadah::clear_staged_extern(); }
  StagedPotential(const StagedPotential &) = delete;
  StagedPotential &operator=(const StagedPotential &) = delete;
};

/* Runs @p lines one at a time with @p pot staged: "" when every line ran,
 * else the failing line and its message. */
std::string execute(const LMPPtr &lmp, const std::vector<std::string> &lines,
                    const Context &pot) {
  StagedPotential staged(pot);
  for (const std::string &line : lines) {
    try {
      lmp->input->one(line);
    } catch (const std::exception &e) {
      return "'" + line + "': " + e.what();
    }
  }
  return "";
}

int lammps_type(const Atom &a) { return a.Z == kTa ? 1 : 2; }

/* An atomic cell (orthogonal box), the atoms in order (ids 1..N), type 1
 * Ta and type 2 Nb (with `two_types` false: Ta only), pair_style tadah,
 * variables of the energy and of every force component; `run 0` follows. */
std::vector<std::string> cell_script(const Structure &st,
                                     bool two_types = true) {
  std::vector<std::string> s = {
      "clear", "units metal", "boundary p p p", "atom_style atomic",
      "atom_modify map array",
      "region box block 0 " + num(st.cell(0, 0)) + " 0 " + num(st.cell(1, 1)) +
          " 0 " + num(st.cell(2, 2)) + " units box",
      two_types ? "create_box 2 box" : "create_box 1 box"};
  for (const Atom &a : st.atoms)
    s.push_back("create_atoms " + std::to_string(lammps_type(a)) +
                " single " + num(a.position[0]) + " " + num(a.position[1]) +
                " " + num(a.position[2]) + " units box remap yes");
  s.push_back("mass 1 180.94788");
  if (two_types) s.push_back("mass 2 92.90637");
  s.push_back("pair_style tadah");
  s.push_back(two_types ? "pair_coeff * * EXTERN Ta Nb"
                        : "pair_coeff * * EXTERN Ta");
  s.push_back("neighbor 1.0 nsq");
  s.push_back("variable e equal pe");
  s.push_back("variable n equal atoms");
  for (std::size_t i = 1; i <= st.natoms(); ++i)
    for (const char *c : {"fx", "fy", "fz"})
      s.push_back(std::string("variable ") + c + std::to_string(i) +
                  " equal " + c + "[" + std::to_string(i) + "]");
  return s;
}

/* The flags PairTadah::compute() dispatches on. */
void require_kernel(const LMPPtr &lmp, bool linear, bool initmb, bool dimer) {
  auto *pt = dynamic_cast<LAMMPS_NS::PairTadah *>(lmp->force->pair);
  REQUIRE(pt != nullptr);
  REQUIRE(pt->lammps_tadah() != nullptr);
  REQUIRE(pt->lammps_tadah()->linear == linear);
  REQUIRE(pt->lammps_tadah()->initmb == initmb);
  REQUIRE(pt->lammps_tadah()->dimer == dimer);
  REQUIRE_FALSE(pt->lammps_tadah()->eam_fast);
}

/* pair_style tadah against the MLIP prediction of @p pot on @p st. */
void compare_with_lammps(const Context &pot, const Structure &st, bool linear,
                         bool initmb, bool two_types = true) {
  MlipPotential mlip(pot, true);
  const Structure pred = mlip.predict(st);
  REQUIRE(pred.atoms.size() == st.natoms());

  LMPPtr lmp = make_instance();
  REQUIRE(lmp != nullptr);
  const std::string err = execute(lmp, cell_script(st, two_types), pot);
  INFO("LAMMPS setup: " << err);
  REQUIRE(err.empty());
  require_kernel(lmp, linear, initmb, false);
  lmp->input->one("run 0 post no");
  LammpsRunner runner;
  REQUIRE(std::stoi(runner.get_variable(lmp, "n")) ==
          static_cast<int>(st.natoms()));
  const double e_lmp = std::stod(runner.get_variable(lmp, "e"));
  INFO("MLIP energy " << num(pred.energy) << ", LAMMPS energy "
                      << num(e_lmp));
  REQUIRE(std::fabs(e_lmp) > 1.0);
  CHECK(pred.energy == Approx(e_lmp).epsilon(1e-9));
  double max_df = 0.0;
  for (std::size_t i = 0; i < st.natoms(); ++i) {
    const std::string id = std::to_string(i + 1);
    const double f[3] = {std::stod(runner.get_variable(lmp, "fx" + id)),
                         std::stod(runner.get_variable(lmp, "fy" + id)),
                         std::stod(runner.get_variable(lmp, "fz" + id))};
    for (int k = 0; k < 3; ++k)
      max_df = std::max(max_df, std::fabs(pred.atoms[i].force[k] - f[k]));
  }
  INFO("largest force difference " << max_df << " eV/A");
  CHECK(max_df <= 1e-8);
}

const std::vector<int> kVaried = {0, 1, 3, 5, 8, 10, 12, 16, 2, 7, 9, 4,
                                  14, 6};

}  // namespace

// Pins: a potential trained in the test with ATOM Nb Ta and BIAS true
// (M_BLR BF_Linear, D2_BP alone and D2_BP + DM_EAD, NORM false and true, on
// cells of varied composition) has one weight per intercept and descriptor
// (5 and 9), and pair_style tadah (compute_2b and compute_2b_mb_half) gives
// the MLIP prediction on a 7-Nb/9-Ta cell: the total energy to 1e-9
// relative, every force to 1e-8 eV/A.
// Before the fix: the potentials had 1 + 3 and 1 + 7 weights (one shared
// intercept), and LAMMPS, which builds 2 + 3 and 2 + 7 descriptor entries,
// gave other energies and forces: the weight-count REQUIRE failed (4 != 5).
TEST_CASE("pair_style tadah and the MLIP prediction agree for a two-species "
          "BIAS potential trained in the test",
          "[multispecies_bias][lammps_oracle]") {
  PeriodicTable::initialize();
  MLIPWorkspaceManager ws;
  const Structure probe = tanb_cell(7, 4242);
  for (int mb = 0; mb < 2; ++mb) {
    for (int norm = 0; norm < 2; ++norm) {
      DYNAMIC_SECTION((mb ? "D2_BP + DM_EAD (compute_2b_mb_half)"
                          : "D2_BP (compute_2b)")
                      << (norm ? ", NORM true" : ", NORM false")) {
        StructureDB db = make_db(kVaried, mb, 50 + 10 * mb);
        Context c = parse(nbta_lines(mb));
        set(c, "NORM", norm ? "true" : "false");
        Trainer tr(c, ws);
        tr.train(db);
        const Context pot = reload(tr.get_param_file());
        REQUIRE(pot.size("WEIGHTS") == (mb ? 9u : 5u));
        compare_with_lammps(pot, probe, true, mb);
      }
    }
  }
}

// Pins: an ATOM Nb Ta BIAS true BF_Polynomial2 potential (D2_BP + DM_EAD,
// 9 descriptor entries, 45 weights, written here) is evaluated alike by
// pair_style tadah (compute_2b_mb_full) and the MLIP prediction (energy to
// 1e-9 relative, forces to 1e-8 eV/A); an ATOM Ta potential of the same
// form (8 entries, 36 weights) is the one-species control.
// Before the fix: the MLIP descriptor had 1 + 7 entries, so the MLIP
// prediction used the first 36 of the 45 weights on another layout: the
// energies differed (MLIP -623.66 eV, LAMMPS -62.51 eV); the one-species
// control died with SIGSEGV in compute_2b_mb_full, whose intercept loop
// read atom->type past nlocal + nghost (exit 139; the same with the MLIP
// fix alone, before pair_tadah's loop bound).
TEST_CASE("pair_style tadah compute_2b_mb_full and the MLIP prediction agree "
          "for a two-species BIAS BF_Polynomial2 potential",
          "[multispecies_bias][lammps_oracle]") {
  PeriodicTable::initialize();
  auto poly2 = [](bool two_species) {
    std::vector<std::string> l = nbta_lines(true);
    if (!two_species) {
      for (auto &x : l)
        if (x == "ATOM Nb Ta") x = "ATOM Ta";
    }
    for (auto &x : l)
      if (x == "MODEL M_BLR BF_Linear") x = "MODEL M_BLR BF_Polynomial2";
    const std::size_t d = two_species ? 9 : 8;
    std::string w = "WEIGHTS";
    std::size_t b = 0;
    for (std::size_t i = 0; i < d; ++i)
      for (std::size_t ii = i; ii < d; ++ii, ++b)
        w += " " + num(i == ii && i < (two_species ? 2u : 1u)
                           ? -7.5 - 0.6 * double(i)
                           : 0.02 * std::sin(1.0 + 0.9 * double(b)));
    l.push_back(w);
    return parse(l);
  };
  SECTION("ATOM Nb Ta") {
    compare_with_lammps(poly2(true), tanb_cell(6, 777), false, true);
  }
  SECTION("ATOM Ta (control)") {
    compare_with_lammps(poly2(false), tanb_cell(0, 778), false, true, false);
  }
}

namespace {

/* ATOM Nb Ta, BIAS true, D2_BP + DM_EAD (2 + 7 descriptor entries) with
 * written weights: M_BLR BF_Linear (9 weights; compute_2b_mb_half) or
 * M_BLR BF_Polynomial2 (45 weights; compute_2b_mb_full). */
Context written_potential(bool poly2) {
  std::vector<std::string> l = nbta_lines(true);
  const std::size_t d = 9;
  std::string w = "WEIGHTS";
  if (poly2) {
    for (auto &x : l)
      if (x == "MODEL M_BLR BF_Linear") x = "MODEL M_BLR BF_Polynomial2";
    std::size_t b = 0;
    for (std::size_t i = 0; i < d; ++i)
      for (std::size_t ii = i; ii < d; ++ii, ++b)
        w += " " + num(i == ii && i < 2 ? -7.5 - 0.6 * double(i)
                                         : 0.02 * std::sin(1.0 + 0.9 * double(b)));
  } else {
    for (std::size_t i = 0; i < d; ++i)
      w += " " + num(i < 2 ? -7.5 - 0.6 * double(i) : 0.03 * std::cos(double(i)));
  }
  l.push_back(w);
  return parse(l);
}

}  // namespace

// Pins: pair_style tadah sets the intercept (bias) components of
// LammpsTadah::aeds for the atoms LAMMPS has, local and ghost, and for no
// other column of the array (it has atom->nmax columns): after a first
// `run 0` on a 6-Nb/10-Ta cell, every atom->type slot past nlocal + nghost
// is set to 1 (Ta, a valid type, as stale memory may hold), and after a
// second `run 0` (the same atoms and ghosts) every entry of aeds in the
// columns past nlocal + nghost is 0, for compute_2b_mb_half (BF_Linear,
// linear with a many-body descriptor) and compute_2b_mb_full
// (BF_Polynomial2); the array has more columns than atoms, so the check
// covers some.
// Before the fix (pair_tadah's loops run to aeds.cols() again, in a scratch
// build): in the compute_2b_mb_half section every column past the atoms
// held a 1 in the Ta intercept row (nonzero_past_atoms 16043: 16384
// columns, 341 atoms), whatever that memory had held; the
// compute_2b_mb_full section, the second LAMMPS instance of the process,
// died with SIGSEGV in its first `run 0`, before the types were set (the
// old loop read unset types there: a crash or a stray write, depending on
// the memory).
TEST_CASE("pair_style tadah sets the intercept components only for the "
          "atoms LAMMPS has, whatever atom->type holds past them",
          "[multispecies_bias][lammps_oracle][bias_loops]") {
  PeriodicTable::initialize();
  for (int poly2 = 0; poly2 < 2; ++poly2) {
    DYNAMIC_SECTION((poly2 ? "compute_2b_mb_full (BF_Polynomial2)"
                           : "compute_2b_mb_half (BF_Linear)")) {
      const Context pot = written_potential(poly2);
      const Structure st = tanb_cell(6, 9191);
      LMPPtr lmp = make_instance();
      REQUIRE(lmp != nullptr);
      const std::string err = execute(lmp, cell_script(st), pot);
      INFO("LAMMPS setup: " << err);
      REQUIRE(err.empty());
      require_kernel(lmp, !poly2, true, false);
      lmp->input->one("run 0 post no");
      LAMMPS_NS::Atom *atom = lmp->atom;
      const int nall = atom->nlocal + atom->nghost;
      REQUIRE(atom->nlocal == static_cast<int>(st.natoms()));
      for (int n = nall; n < atom->nmax; ++n) atom->type[n] = 1;
      lmp->input->one("run 0 post no");
      REQUIRE(atom->nlocal + atom->nghost == nall);
      auto *pt = dynamic_cast<LAMMPS_NS::PairTadah *>(lmp->force->pair);
      REQUIRE(pt != nullptr);
      const tadah::LammpsTadah *lt = pt->lammps_tadah();
      REQUIRE(lt != nullptr);
      const std::size_t cols = lt->aeds.cols();
      INFO("aeds columns " << cols << ", atoms (nlocal + nghost) " << nall
                           << ", atom->nmax " << atom->nmax);
      REQUIRE(cols > static_cast<std::size_t>(nall));
      std::size_t nonzero_past_atoms = 0;
      for (std::size_t n = static_cast<std::size_t>(nall); n < cols; ++n)
        for (std::size_t k = 0; k < lt->aeds.rows(); ++k)
          if (lt->aeds(k, n) != 0.0) ++nonzero_past_atoms;
      CHECK(nonzero_past_atoms == 0);
    }
  }
}

namespace {

/* A DIMER BF_Polynomial2 potential: ATOM Nb Ta, BIAS true, D2_BP (3) for
 * every pair, molecules with bond 1.1 A; 2 + 3 entries, 15 weights. */
Context dimer_potential() {
  std::vector<std::string> l = nbta_lines(false);
  for (auto &x : l) {
    if (x == "DIMER false 0 false") x = "DIMER true 1.1 true";
    if (x == "MODEL M_BLR BF_Linear") x = "MODEL M_BLR BF_Polynomial2";
  }
  std::string w = "WEIGHTS";
  std::size_t b = 0;
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t ii = i; ii < 5; ++ii, ++b)
      w += " " + num(i == ii && i < 2 ? -3.0 - 0.8 * double(i)
                                       : 0.05 * std::sin(0.3 + 1.1 * double(b)));
  l.push_back(w);
  return parse(l);
}

}  // namespace

// Pins: pair_style tadah evaluates a two-species BIAS DIMER potential
// (compute_dimers) as the MLIP prediction does (energy to 1e-9 relative):
// two Nb-Ta molecules whose atoms are created interleaved (ids 1 = A Nb,
// 2 = B Nb, 3 = A Ta, 4 = B Ta; bonds 1-3 and 2-4) and not sorted
// (atom_modify sort 0), so the molecule pair compute_dimers forms, i1 i2
// j1 j2 = local atoms 0 2 1 3, is not the order of the local atoms, and
// each of its four descriptors must take the intercept of its own atom's
// species.
// Before the fix: the MLIP descriptor had 1 + 3 entries and used the first
// 10 of the 15 weights (MLIP -15.402 against LAMMPS -13.574 eV); with that
// fixed (MLIP -13.552 eV), compute_dimers still set the intercept of
// descriptor n from the type of local atom n (Nb Nb Ta Ta for the atoms
// Nb Ta Nb Ta of the pair).
TEST_CASE("pair_style tadah compute_dimers and the MLIP prediction agree for "
          "a two-species BIAS DIMER potential",
          "[multispecies_bias][lammps_oracle][dimer]") {
  PeriodicTable::initialize();
  const Context pot = dimer_potential();
  // Molecule A along x from (10, 10, 10); molecule B 3.4 A further along x,
  // tilted.
  const double h = 0.55;
  const double u[3] = {0.3, 0.1, std::sqrt(1.0 - 0.09 - 0.01)};
  const double ca[3] = {10.55, 10.0, 10.0};
  const double cb[3] = {13.95, 10.0, 10.0};
  double a_nb[3], a_ta[3], b_nb[3], b_ta[3];
  for (int k = 0; k < 3; ++k) {
    a_nb[k] = ca[k] - (k == 0 ? h : 0.0);
    a_ta[k] = ca[k] + (k == 0 ? h : 0.0);
    b_nb[k] = cb[k] - h * u[k];
    b_ta[k] = cb[k] + h * u[k];
  }

  // MLIP: molecule I = atoms 0, 1; molecule J = atoms 2, 3.
  Structure st;
  st.cell.load(30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 30.0);
  const auto Nb = PeriodicTable::find_by_symbol("Nb");
  const auto Ta = PeriodicTable::find_by_symbol("Ta");
  st.add_atom(Atom(Nb, a_nb[0], a_nb[1], a_nb[2], 0, 0, 0));
  st.add_atom(Atom(Ta, a_ta[0], a_ta[1], a_ta[2], 0, 0, 0));
  st.add_atom(Atom(Nb, b_nb[0], b_nb[1], b_nb[2], 0, 0, 0));
  st.add_atom(Atom(Ta, b_ta[0], b_ta[1], b_ta[2], 0, 0, 0));
  MlipPotential mlip(pot, false);
  const double e_mlip = mlip.predict(st).energy;

  // LAMMPS: ids 1 = A Nb, 2 = B Nb, 3 = A Ta, 4 = B Ta (type 1 Ta, 2 Nb).
  auto at = [](int type, const double *r) {
    return "create_atoms " + std::to_string(type) + " single " + num(r[0]) +
           " " + num(r[1]) + " " + num(r[2]) + " units box";
  };
  const std::vector<std::string> script = {
      "clear", "units metal", "boundary p p p", "newton off",
      "atom_style bond", "atom_modify map array sort 0 0.0",
      "region box block 0 30 0 30 0 30 units box",
      "create_box 2 box bond/types 1 extra/bond/per/atom 1 "
      "extra/special/per/atom 1",
      at(2, a_nb), at(2, b_nb), at(1, a_ta), at(1, b_ta),
      "mass 1 180.94788", "mass 2 92.90637", "bond_style zero",
      "bond_coeff 1 1.1", "create_bonds single/bond 1 1 3",
      "create_bonds single/bond 1 2 4", "pair_style tadah",
      "pair_coeff * * EXTERN Ta Nb", "neighbor 1.0 nsq",
      "variable e equal pe"};
  LMPPtr lmp = make_instance();
  REQUIRE(lmp != nullptr);
  const std::string err = execute(lmp, script, pot);
  INFO("LAMMPS setup: " << err);
  REQUIRE(err.empty());
  require_kernel(lmp, false, false, true);
  lmp->input->one("run 0 post no");
  const double e_lmp = std::stod(LammpsRunner().get_variable(lmp, "e"));
  INFO("MLIP energy " << num(e_mlip) << ", LAMMPS energy " << num(e_lmp));
  REQUIRE(std::fabs(e_lmp) > 1.0);
  CHECK(e_mlip == Approx(e_lmp).epsilon(1e-9));
}

// Pins: pair_coeff refuses an ATOM Nb Ta BIAS true potential with 1 + 3
// WEIGHTS (one intercept column shared by the two species, as tadah trained
// it before 1.6.0) for its 2 + 3 descriptor entries, naming WEIGHTS,
// DSIZE, BIAS and ATOM ("has 4 WEIGHTS", "5 components (DSIZE)"), and saying
// to retrain.
// Before the fix: pair_coeff accepted it (setup error empty). Before the
// one-guard rewording (review): the message said "WEIGHTS has 4 values" and
// "DSIZE 5".
TEST_CASE("pair_coeff refuses a two-species BIAS potential with one shared "
          "intercept column",
          "[multispecies_bias][lammps_oracle]") {
  PeriodicTable::initialize();
  std::vector<std::string> l = nbta_lines(false);
  l.push_back("WEIGHTS -8.2 0.1 -0.05 0.02");
  const Context pot = parse(l);
  LMPPtr lmp = make_instance();
  REQUIRE(lmp != nullptr);
  const std::string err = execute(lmp, cell_script(tanb_cell(5, 31)), pot);
  INFO(err);
  CHECK(err.find("pair_coeff") != std::string::npos);
  CHECK(err.find("has 4 WEIGHTS") != std::string::npos);
  CHECK(err.find("5 components (DSIZE)") != std::string::npos);
  CHECK(err.find("one intercept column shared by all species") !=
        std::string::npos);
  CHECK(err.find("retrain") != std::string::npos);
}

// Pins: the model weight count tadah hpo derives from a -c seed that is
// such a potential (1 + 3 stale WEIGHTS) is that of the per-species layout,
// 2 + 3: the stale WEIGHTS are not read (the fit replaces them).
// Before the fix: 4 (one shared intercept).
TEST_CASE("derive_weight_count: a two-species BIAS seed with stale WEIGHTS "
          "has one weight per intercept and descriptor",
          "[multispecies_bias][startup]") {
  PeriodicTable::initialize();
  std::vector<std::string> l = nbta_lines(false);
  l.push_back("WEIGHTS -8.2 0.1 -0.05 0.02");
  std::size_t k = 0;
  REQUIRE_NOTHROW(k = tadah::hpo::derive_weight_count(parse(l)));
  CHECK(k == 5);
}

namespace {

struct ScopedFile {
  fs::path path;
  explicit ScopedFile(const std::string &body, const std::string &ext = ".in")
      : path(fs::temp_directory_path() /
             ("tadah_hpo_msbias_" +
              std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()) +
              ext)) {
    std::ofstream os(path);
    os << body;
  }
  ~ScopedFile() {
    std::error_code ec;
    fs::remove(path, ec);
  }
};

const char *kHpoTarget =
    "OPTIMIZER\nLIB DLIB\nALGO BFGS\nMAXEVAL 1\nENDOPTIMIZER\n"
    "LOSS L2\nPC_EMAE 0 1\nOPTIM CGRID2B (1) 2.0 2.8\n";

/* Three atoms (Nb or Ta) of a small cluster, shifted by `dx`. */
Structure cluster3(const std::vector<std::string> &el, double dx) {
  Structure st;
  st.cell.load(20.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 20.0);
  const double p[3][3] = {{0.0, 0.0, 0.0}, {2.6, 0.0, 0.0}, {0.3, 2.9, 0.0}};
  for (int a = 0; a < 3; ++a)
    st.add_atom(Atom(PeriodicTable::find_by_symbol(el[a]),
                     5.0 + p[a][0] + (a == 1 ? dx : 0.0), 5.0 + p[a][1],
                     5.0 + p[a][2], 0, 0, 0));
  return st;
}

}  // namespace

// Pins: HPO's analytical gradient of an EMAE loss with respect to CGRID2B
// (1) for an ATOM Nb Ta BIAS true model (D2_BP, OALGO 4, LAMBDA 1e-3) on
// three-atom clusters of every composition matches central differences of
// the loss over re-fits (1e-3 relative): the gradient builds its design
// matrix with the DC_Selector width, which is the trainer's DSIZE (2 + 3),
// the size of the descriptors it reads.
// Before the fix: DSIZE 4 against the DC_Selector's 5 (the REQUIRE
// failed); the gradient read 5 columns of each atom's 4-entry descriptor,
// past its end (in a run of this case alone: analytical inf against
// finite differences -729; a run of all the cases read other memory).
TEST_CASE("HPO analytical gradient of a two-species BIAS model matches "
          "finite differences",
          "[multispecies_bias][analytical_gradient]") {
  PeriodicTable::initialize();
  const std::vector<std::vector<std::string>> comps = {
      {"Nb", "Nb", "Nb"}, {"Nb", "Nb", "Ta"}, {"Nb", "Ta", "Ta"},
      {"Ta", "Ta", "Ta"}, {"Ta", "Nb", "Ta"}, {"Ta", "Nb", "Nb"}};
  StructureDB train, val;
  for (std::size_t i = 0; i < comps.size(); ++i) {
    Structure st = cluster3(comps[i], 0.05 * double(i));
    st.energy = -8.0 * 3 + 0.04 * double(i + 1) - 0.3 * count_of(st, kNb);
    train.add(st);
  }
  for (std::size_t i = 0; i < 3; ++i) {
    Structure st = cluster3(comps[2 * i + 1], 0.13 + 0.03 * double(i));
    st.energy = -8.0 * 3 + 0.06 * double(i + 1) - 0.3 * count_of(st, kNb);
    val.add(st);
  }
  train.dbidx = {0, train.size()};
  val.dbidx = {0, val.size()};
  NNFinder().calc(train, kRcut);
  NNFinder().calc(val, kRcut);

  auto context = [](double cgrid1) {
    std::vector<std::string> l = nbta_lines(false);
    for (auto &x : l) {
      if (x == "CGRID2B 2.4 3.1 3.9") x = "CGRID2B " + num(cgrid1) + " 3.1 3.9";
      if (x == "LAMBDA 0") x = "LAMBDA 1e-3";
      if (x == "OALGO 1") x = "OALGO 4";
    }
    return parse(l);
  };
  const double theta0 = 2.4;
  ScopedFile cfg(kHpoTarget);
  Context ctx_trg = context(theta0);
  tadah::hpo::HPOTargets trg(ctx_trg, cfg.path.string());
  const auto prefs = tadah::hpo::make_param_ref_table(trg);
  REQUIRE(prefs.size() == 1);

  MLIPWorkspaceManager ws;
  Context c0 = context(theta0);
  Trainer host(c0, ws);
  REQUIRE(host.context.get<std::size_t>("DSIZE") ==
          host.DCS.descriptor_size());
  host.train(train);
  std::vector<double> g;
  std::vector<int> needs_fd;
  REQUIRE(tadah::hpo::compute_analytical_gradient_from_trainer(
      host, train, val, trg, prefs, g, needs_fd));
  REQUIRE(needs_fd.empty());
  REQUIRE(g.size() == 1);

  auto loss_at = [&](double theta) {
    MLIPWorkspaceManager w;
    Context c = context(theta);
    Trainer t(c, w);
    t.train(train);
    double sum = 0.0;
    for (std::size_t s = 0; s < val.size(); ++s) {
      const StDescriptors d = t.dc.calc(val(s));
      sum += std::fabs(val(s).energy - t.model->epredict(d)) /
             double(val(s).natoms());
    }
    const double emae = 1000.0 * sum / double(val.size());
    return trg.globalLoss.loss(emae, trg.emae, trg.w_emae);
  };
  const double step = 1e-5;
  const double fd = (loss_at(theta0 + step) - loss_at(theta0 - step)) /
                    (2.0 * step);
  INFO("analytical " << g[0] << ", finite differences " << fd);
  const double rel = std::fabs(g[0] - fd) /
                     std::max({std::fabs(g[0]), std::fabs(fd), 1e-12});
  CHECK(rel < 1e-3);
}

// Pins: tadah hpo's startup validation (validate_hpo_setup) refuses NORM
// true with M_KRR Kern_Linear and BIAS false (as tadah train does), and
// warns, on standard error, that the descriptor weights start at index 3
// when FIXINDEX is used with BIAS true and two species.
// Before the fix: neither (no exception, no warning).
TEST_CASE("validate_hpo_setup: NORM with Kern_Linear needs BIAS; FIXINDEX "
          "with two-species BIAS warns about the index shift",
          "[multispecies_bias][startup]") {
  PeriodicTable::initialize();
  ScopedFile cfg(kHpoTarget);
  SECTION("NORM true, M_KRR Kern_Linear, BIAS false") {
    std::vector<std::string> l = nbta_lines(false, false);
    for (auto &x : l) {
      if (x == "NORM false") x = "NORM true";
      if (x == "MODEL M_BLR BF_Linear") x = "MODEL M_KRR Kern_Linear 1";
    }
    Context ctx = parse(l);
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    CHECK_THROWS_WITH(tadah::hpo::validate_hpo_setup(ctx, trg, true, true),
                      Catch::Contains("NORM") && Catch::Contains("BIAS"));
  }
  SECTION("FIXINDEX, BIAS true, ATOM Nb Ta") {
    std::vector<std::string> l = nbta_lines(false);
    l.push_back("DBFILE train.db");   // not read by the startup validation
    l.push_back("FIXINDEX 3");
    l.push_back("FIXWEIGHT 0.1");
    Context ctx = parse(l);
    tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
    std::ostringstream captured;
    std::streambuf *old = std::cerr.rdbuf(captured.rdbuf());
    try {
      tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
    } catch (...) {
      std::cerr.rdbuf(old);
      throw;
    }
    std::cerr.rdbuf(old);
    INFO(captured.str());
    CHECK(captured.str().find("FIXINDEX with BIAS true") != std::string::npos);
    CHECK(captured.str().find("start at index 3") != std::string::npos);
  }
}

namespace {

/* @p structures in Tadah's dataset format: structure s labelled "cell s",
 * energy -130 - s, no forces or stresses. */
std::string dataset_of(const std::vector<Structure> &structures) {
  std::ostringstream db;
  for (std::size_t s = 0; s < structures.size(); ++s) {
    const Structure &st = structures[s];
    db << "cell " << s << "\n1.0 1.0 1.0\n" << num(-130.0 - double(s))
       << " 0.0\n";
    for (int r = 0; r < 3; ++r)
      db << num(st.cell(r, 0)) << " " << num(st.cell(r, 1)) << " "
         << num(st.cell(r, 2)) << "\n";
    db << "0 0 0\n0 0 0\n0 0 0\n";
    for (const Atom &a : st.atoms)
      db << a.symbol << " " << num(a.position[0]) << " "
         << num(a.position[1]) << " " << num(a.position[2])
         << " 0 0 0\n";
    db << "\n";
  }
  return db.str();
}

}  // namespace

// Pins: tadah hpo refuses, when it loads the training data (HPO_Host),
// BIAS true data whose energy-weighted compositions cannot determine the
// per-species intercepts (every structure 8 Nb + 8 Ta), naming the rank
// and the remedies.
// Before the fix: the host was built; every evaluation would fit an
// arbitrary split of the energy between the species.
TEST_CASE("tadah hpo refuses BIAS true training data of one composition",
          "[multispecies_bias][startup]") {
  using ParamVec = dlib::matrix<double, 0, 1>;
  using Host = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;
  PeriodicTable::initialize();
  Host::reset_globals_for_test();
  // Two structures of 8 Nb + 8 Ta in Tadah's dataset format.
  ScopedFile train_db(dataset_of({tanb_cell(8, 90), tanb_cell(8, 91)}),
                      ".db");
  ScopedFile val_cfg("DBFILE " + train_db.path.string() + "\n");
  ScopedFile cfg(kHpoTarget);
  std::vector<std::string> l = nbta_lines(false);
  l.push_back("DBFILE " + train_db.path.string());
  Context ctx = parse(l);
  tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
  std::string m = "<no exception>";
  try {
    Host host(ctx, trg, val_cfg.path.string());
  } catch (const std::exception &e) {
    m = e.what();
  }
  INFO(m);
  CHECK(m.find("cannot determine") != std::string::npos);
  CHECK(m.find("rank 1") != std::string::npos);
  CHECK(m.find("Nb8 Ta8") != std::string::npos);
}

namespace {

using ParamVec = dlib::matrix<double, 0, 1>;
using MsbHost = tadah::hpo::HPO_Host<ParamVec, ::LammpsRunner>;

/* nbta_lines (ATOM Nb Ta, BIAS true, D2_BP: DSIZE 5) with MODEL @p model
 * and the lines @p extra. */
std::vector<std::string> model_lines(const std::string &model,
                                     const std::vector<std::string> &extra) {
  std::vector<std::string> l = nbta_lines(false);
  for (auto &x : l)
    if (x == "MODEL M_BLR BF_Linear") x = "MODEL " + model;
  l.insert(l.end(), extra.begin(), extra.end());
  return l;
}

/* An HPO_Host built from @p lines and the training structures @p train
 * (also the validation set): "<built>", or the message of its refusal. */
std::string host_error(std::vector<std::string> lines,
                       const std::vector<Structure> &train) {
  MsbHost::reset_globals_for_test();
  ScopedFile train_db(dataset_of(train), ".db");
  ScopedFile val_cfg("DBFILE " + train_db.path.string() + "\n");
  ScopedFile cfg(kHpoTarget);
  lines.push_back("DBFILE " + train_db.path.string());
  Context ctx = parse(lines);
  tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
  try {
    MsbHost host(ctx, trg, val_cfg.path.string());
  } catch (const std::exception &e) {
    return e.what();
  }
  return "<built>";
}

/* What validate_hpo_setup writes to standard error for @p lines (a
 * training DBFILE named, not read), or "<refused> " and its message. */
std::string startup_output(std::vector<std::string> lines) {
  ScopedFile cfg(kHpoTarget);
  lines.push_back("DBFILE train.db");
  Context ctx = parse(lines);
  tadah::hpo::HPOTargets trg(ctx, cfg.path.string());
  std::ostringstream captured;
  std::streambuf *old = std::cerr.rdbuf(captured.rdbuf());
  std::string refused;
  try {
    tadah::hpo::validate_hpo_setup(ctx, trg, true, true);
  } catch (const std::exception &e) {
    refused = std::string("<refused> ") + e.what();
  }
  std::cerr.rdbuf(old);
  return refused.empty() ? captured.str() : refused;
}

/* @p n structures of 8 Nb + 8 Ta. */
std::vector<Structure> one_composition(unsigned n) {
  std::vector<Structure> v;
  for (unsigned s = 0; s < n; ++s) v.push_back(tanb_cell(8, 90 + s));
  return v;
}

/* Structures of 0, 4, 8, 12 and 16 Nb atoms of 16. */
std::vector<Structure> varied_compositions() {
  std::vector<Structure> v;
  for (int k = 0; k < 5; ++k) v.push_back(tanb_cell(4 * k, 70 + k));
  return v;
}

/* BASIS values of the basis vectors e_0, e_1 and @p more vectors of 5
 * values (an intercept component 1 and three descriptor values each), and
 * SBASIS their number. */
std::vector<std::string> basis_lines(int more) {
  std::ostringstream b;
  b << "BASIS 1 0 0 0 0 0 1 0 0 0";
  for (int k = 0; k < more; ++k)
    b << " " << (k % 2 == 0 ? "1 0 " : "0 1 ") << num(0.2 + 0.1 * k) << " "
      << num(0.5 - 0.05 * k) << " " << num(0.1 + 0.02 * k);
  return {"SBASIS " + std::to_string(2 + more), b.str()};
}

}  // namespace

// Pins: tadah hpo's composition check at startup (HPO_Host, when it loads
// the training data) is that of the model's function, built as an
// evaluation builds it, on two structures of 8 Nb + 8 Ta: M_BLR
// BF_Polynomial2 (DSIZE 5, 15 weights) is refused naming "rank 1" and its
// intercept weights "weights 1 (Nb) and 6 (Ta)", the products (0,0) and
// (1,1); with FIXINDEX 6 / FIXWEIGHT -8.9, which pin the Ta constant, the
// host is built; M_KRR Kern_RBF (MPARAMS 1.0, SBASIS 6), whose basis
// vectors are no intercepts, is built.
// Before the fix (HPO before 2fa6344, with MLIP before 0e65254, whose
// check took the weights 1 to N for the intercepts whatever the function;
// this case came after the fix, which had no HPO test): BF_Polynomial2 was
// refused naming "weights 1 to 2 are the intercepts of Nb Ta", with
// FIXINDEX 6 it was refused too ("rank 1, while 2 intercepts are fitted"),
// and Kern_RBF was refused.
TEST_CASE("tadah hpo's startup composition check takes the intercepts of "
          "the model's function",
          "[multispecies_bias][startup]") {
  PeriodicTable::initialize();
  SECTION("M_BLR BF_Polynomial2") {
    const std::string m =
        host_error(model_lines("M_BLR BF_Polynomial2", {}), one_composition(2));
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("rank 1") != std::string::npos);
    CHECK(m.find("weights 1 (Nb) and 6 (Ta)") != std::string::npos);
  }
  SECTION("M_BLR BF_Polynomial2, FIXINDEX 6: the Ta constant pinned") {
    const std::string m = host_error(
        model_lines("M_BLR BF_Polynomial2", {"FIXINDEX 6", "FIXWEIGHT -8.9"}),
        one_composition(2));
    INFO(m);
    CHECK(m == "<built>");
  }
  SECTION("M_KRR Kern_RBF") {
    const std::string m = host_error(
        model_lines("M_KRR Kern_RBF 1", {"MPARAMS 1.0", "SBASIS 6"}),
        one_composition(2));
    INFO(m);
    CHECK(m == "<built>");
  }
}

// Pins: the startup FIXINDEX WARNING (validate_hpo_setup, standard error)
// with BIAS true on ATOM Nb Ta names the intercept weights of the model's
// function: for M_BLR BF_Polynomial2 (FIXINDEX 4) "weights 1 (Nb) and 6
// (Ta)", without the linear models' "start at index 3"; for M_KRR
// Kern_Quadratic (FIXINDEX 3, SBASIS 6) "its intercepts are weights 1 (Nb)
// and 2 (Ta)", the intercept vectors e_0 and e_1 its fit draws first.
// Before the fix: BF_Polynomial2 got the linear models' WARNING, "weights
// 1 to 2 are the per-species intercepts ... the descriptor weights start at
// index 3" (HPO before 2fa6344, with MLIP before 0e65254); Kern_Quadratic
// got "weights 1 to 2 belong to its intercept basis vectors e_0 .. e_1"
// without its intercepts (the probe's kernel has no basis before its
// fit).
TEST_CASE("validate_hpo_setup: the FIXINDEX WARNING names the intercept "
          "weights of the model's function",
          "[multispecies_bias][startup]") {
  PeriodicTable::initialize();
  SECTION("M_BLR BF_Polynomial2") {
    const std::string out = startup_output(model_lines(
        "M_BLR BF_Polynomial2", {"FIXINDEX 4", "FIXWEIGHT 0.05"}));
    INFO(out);
    CHECK(out.find("FIXINDEX with BIAS true") != std::string::npos);
    CHECK(out.find("weights 1 (Nb) and 6 (Ta)") != std::string::npos);
    CHECK(out.find("start at index 3") == std::string::npos);
  }
  SECTION("M_KRR Kern_Quadratic") {
    const std::string out = startup_output(model_lines(
        "M_KRR Kern_Quadratic 1", {"SBASIS 6", "FIXINDEX 3", "FIXWEIGHT 0.05"}));
    INFO(out);
    CHECK(out.find("FIXINDEX with BIAS true") != std::string::npos);
    CHECK(out.find("its intercepts are weights 1 (Nb) and 2 (Ta)") !=
          std::string::npos);
  }
}

// Pins: tadah hpo checks a non-linear kernel's intercepts at startup
// (HPO_Host) on the basis its fit draws, whose first vectors are the
// intercept vectors e_0 and e_1, and reads no basis of a -c potential:
// M_KRR Kern_Quadratic 1 (SBASIS 6) on four structures of 8 Nb + 8 Ta is
// refused naming "rank 1" and "weights 1 (Nb) and 2 (Ta)"; M_KRR
// Kern_Quadratic 2 (SBASIS 5), whose energies-only solve is not checked,
// builds the host on the same data, from a training configuration and from
// a potential (BASIS e_0, e_1 and three more vectors, 5 WEIGHTS); a
// Kern_Quadratic 1 potential of SBASIS 6 (30 BASIS values, 6 WEIGHTS) with
// SBASIS 7, a seed whose basis the fit redraws, builds it on structures of
// five compositions.
// Before the fix: Kern_Quadratic 1 passed startup (the probe's kernel has
// no basis before its fit, so nothing was checked), and every evaluation
// failed ("cannot determine"); the Kern_Quadratic 2 potential was refused,
// "rank 1", the intercepts taken from its basis e_0, e_1; the SBASIS 7 seed
// was refused, "BASIS has 30 values, which is not a multiple of SBASIS
// 7".
TEST_CASE("tadah hpo checks a non-linear kernel's intercepts at startup on "
          "the basis its fit draws",
          "[multispecies_bias][startup]") {
  PeriodicTable::initialize();
  SECTION("Kern_Quadratic 1, one composition") {
    const std::string m = host_error(
        model_lines("M_KRR Kern_Quadratic 1", {"SBASIS 6"}),
        one_composition(4));
    INFO(m);
    CHECK(m.find("cannot determine") != std::string::npos);
    CHECK(m.find("rank 1") != std::string::npos);
    CHECK(m.find("weights 1 (Nb) and 2 (Ta)") != std::string::npos);
  }
  SECTION("Kern_Quadratic 2, one composition") {
    const std::string m = host_error(
        model_lines("M_KRR Kern_Quadratic 2", {"SBASIS 5"}),
        one_composition(4));
    INFO(m);
    CHECK(m == "<built>");
  }
  SECTION("Kern_Quadratic 2 potential, one composition") {
    std::vector<std::string> extra = basis_lines(3);
    extra.push_back("WEIGHTS -4.1 -4.4 0.3 -0.2 0.1");
    const std::string m = host_error(
        model_lines("M_KRR Kern_Quadratic 2", extra), one_composition(4));
    INFO(m);
    CHECK(m == "<built>");
  }
  SECTION("Kern_Quadratic 1 potential of SBASIS 6 seeding SBASIS 7") {
    std::vector<std::string> extra = basis_lines(4);
    extra[0] = "SBASIS 7";
    extra.push_back("WEIGHTS -4.1 -4.4 0.3 -0.2 0.1 0.05");
    const std::string m = host_error(
        model_lines("M_KRR Kern_Quadratic 1", extra), varied_compositions());
    INFO(m);
    CHECK(m == "<built>");
  }
}

// Pins: tadah hpo's startup (validate_hpo_setup) refuses SBASIS 2 for
// M_KRR Kern_RBF with BIAS true on ATOM Nb Ta, naming "does not build a
// model", "SBASIS 2", "at least 3 basis vectors" and "the 2 intercept
// vectors" (with one drawn vector, the fewest a fit takes); SBASIS 3
// passes.
// Before the fix: SBASIS 2 passed startup (the bound was 2 whatever the
// species), and every evaluation failed ("needs at least 3 basis
// vectors").
TEST_CASE("validate_hpo_setup: SBASIS of a non-linear kernel with BIAS true "
          "on two species counts one intercept vector per species",
          "[multispecies_bias][startup]") {
  PeriodicTable::initialize();
  {
    const std::string out = startup_output(
        model_lines("M_KRR Kern_RBF 1", {"MPARAMS 1.0", "SBASIS 2"}));
    INFO(out);
    CHECK(out.find("<refused>") == 0);
    CHECK(out.find("does not build a model") != std::string::npos);
    CHECK(out.find("SBASIS 2") != std::string::npos);
    CHECK(out.find("at least 3 basis vectors") != std::string::npos);
    CHECK(out.find("the 2 intercept vectors") != std::string::npos);
  }
  const std::string out = startup_output(
      model_lines("M_KRR Kern_RBF 1", {"MPARAMS 1.0", "SBASIS 3"}));
  INFO(out);
  CHECK(out.find("<refused>") == std::string::npos);
}

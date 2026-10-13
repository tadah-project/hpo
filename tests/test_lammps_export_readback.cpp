// test_lammps_export_readback.cpp
//
// `tadah data write -f LAMMPS` (MLIP LammpsStructureWriter) writes a LAMMPS
// general-triclinic data file. This test reads the written file back into
// an in-process LAMMPS (read_data) and checks what LAMMPS makes of it, which
// the MLIP writer tests (text parsing only) cannot: every atom arrives, the
// box has the volume of the cell, and each atom sits at the box fractions it
// has in the cell. A right-handed cell is written row by row, so its atoms
// keep their fractional coordinates f; a left-handed one (valid in Tadah,
// e.g. read from CASTEP; LAMMPS refuses such a box) is written as a0, a1,
// -a2 from "abc origin" a2, so its atoms are at (f0, f1, 1 - f2). With the
// origin at 0 a left-handed export put the atoms outside the box along C:
// LAMMPS wrapped them only when that direction is periodic and stopped with
// "Did not assign all atoms correctly" under `boundary p p f`.

#include "catch2/catch.hpp"

#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_writers/lammps_structure_writer.h>

#include <atom.h>
#include <domain.h>
#include <input.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using tadah::core::Element;
using tadah::core::Matrix3d;
using tadah::core::PeriodicTable;
using tadah::mlip::Atom;
using tadah::mlip::LammpsStructureWriter;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

const double kFrac[2][3] = {{0.1, 0.2, 0.3}, {0.6, 0.7, 0.0}};

// Two Ta atoms at kFrac of `cell` (the second on the c = 0 face).
Structure two_atoms(const Matrix3d &cell) {
  Structure st;
  st.label = "readback";
  st.cell = cell;
  for (const auto &f : kFrac) {
    double r[3];
    for (int k = 0; k < 3; ++k)
      r[k] = f[0] * cell(0, k) + f[1] * cell(1, k) + f[2] * cell(2, k);
    st.add_atom(Atom(Element("Ta"), r[0], r[1], r[2], 0.0, 0.0, 0.0));
  }
  return st;
}

double abs_det(const Matrix3d &c) {
  return std::fabs(c(0, 0) * (c(1, 1) * c(2, 2) - c(1, 2) * c(2, 1)) -
                   c(0, 1) * (c(1, 0) * c(2, 2) - c(1, 2) * c(2, 0)) +
                   c(0, 2) * (c(1, 0) * c(2, 1) - c(1, 1) * c(2, 0)));
}

struct Readback {
  std::string error;                       // a LAMMPS error, if any
  double natoms = 0;
  double volume = 0;
  std::vector<std::array<double, 3>> lamda;  // by atom ID - 1
};

Readback read_back(const std::string &file, const std::string &boundary) {
  Readback rb;
  LammpsRunner runner;
  auto lmp = runner.create_instance(
      {"lmp", "-log", "none", "-screen", "none", "-nocite"});
  REQUIRE(lmp != nullptr);
  try {
    lmp->input->one("units metal");
    lmp->input->one("atom_style atomic");
    lmp->input->one("atom_modify map array");
    lmp->input->one("boundary " + boundary);
    lmp->input->one("read_data " + file);
  } catch (const std::exception &e) {
    rb.error = e.what();
    return rb;
  }
  rb.natoms = static_cast<double>(lmp->atom->natoms);
  rb.volume = lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd;
  rb.lamda.assign(static_cast<std::size_t>(lmp->atom->natoms), {0, 0, 0});
  for (int i = 0; i < lmp->atom->nlocal; ++i) {
    double lam[3];
    lmp->domain->x2lamda(lmp->atom->x[i], lam);
    const auto id = static_cast<std::size_t>(lmp->atom->tag[i] - 1);
    REQUIRE(id < rb.lamda.size());
    rb.lamda[id] = {lam[0], lam[1], lam[2]};
  }
  return rb;
}

// Box fraction `got` equals `want` modulo 1 (LAMMPS maps periodic
// coordinates into [0, 1); a point on a face may come back as 0 or 1).
void check_fraction(double got, double want) {
  double d = got - want;
  d -= std::round(d);
  CHECK(std::fabs(d) < 1e-10);
}

}  // namespace

TEST_CASE("LAMMPS reads a written data file back: every atom inside the box "
          "at its cell fractions, right- and left-handed, under periodic and "
          "non-periodic boundaries",
          "[lammps_export][cell_convention][cell_handedness]") {
  PeriodicTable::initialize();
  const double s = 4.9343;
  Matrix3d md_stress;  // the md_stress.castep rows: left-handed, det -2 s^3
  md_stress.load(s, s, 0.0,
                 s, 0.0, s,
                 0.0, s, s);
  Matrix3d triclinic;  // right-handed, every entry non-zero
  triclinic.load(4.2, 0.3, -0.2,
                 1.3, 3.9, 0.4,
                 -0.9, 1.1, 4.6);
  struct Case { const char *name; Matrix3d cell; bool left; };
  for (const Case &c : {Case{"right-handed triclinic", triclinic, false},
                        Case{"left-handed md_stress", md_stress, true}}) {
    const std::string file = "lmp_readback_" + std::to_string(::getpid()) +
                             ".data";
    {
      StructureDB db;
      db.add(two_atoms(c.cell));
      LammpsStructureWriter w(db);
      w.set_precision(18);
      w.write_data(file, 0);
    }
    for (const char *boundary : {"p p p", "p p f"}) {
      INFO(c.name << ", boundary " << boundary);
      const Readback rb = read_back(file, boundary);
      REQUIRE(rb.error.empty());
      CHECK(rb.natoms == 2);
      CHECK(rb.volume == Approx(abs_det(c.cell)).epsilon(1e-12));
      REQUIRE(rb.lamda.size() == 2);
      for (std::size_t n = 0; n < 2; ++n) {
        INFO("atom " << n + 1);
        check_fraction(rb.lamda[n][0], kFrac[n][0]);
        check_fraction(rb.lamda[n][1], kFrac[n][1]);
        check_fraction(rb.lamda[n][2],
                       c.left ? 1.0 - kFrac[n][2] : kFrac[n][2]);
      }
    }
    std::remove(file.c_str());
  }
}

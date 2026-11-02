// lammps_styles.h
//
// Whether the LAMMPS library the tests are linked with has a style. A LAMMPS
// build has the styles of the packages it was configured with: atom_style
// bond, with which the DIMER cases make their bonded molecules, is part of
// the MOLECULE package. A test case that needs such a style is skipped by a
// LAMMPS without it.

#pragma once

#include <tadah/hpo/runners.h>

#include <library.h>

#include <stdexcept>
#include <string>

/* True when a LAMMPS instance has the style @p name of @p category ("atom",
 * "pair", "bond", ...), by the LAMMPS library interface. Throws when no
 * instance can be created, so that the test case fails instead of being
 * skipped for a style it was not asked about. */
inline bool lammps_has(const char *category, const char *name) {
  std::string error;
  const LammpsRunner::LMPPtr lmp = LammpsRunner().create_instance(
      {"lmp", "-log", "none", "-screen", "none"}, &error);
  if (!lmp)
    throw std::runtime_error(std::string("no LAMMPS instance to look up the ") +
                             category + " style " + name + " in: " + error);
  return lammps_has_style(lmp.get(), category, name) != 0;
}

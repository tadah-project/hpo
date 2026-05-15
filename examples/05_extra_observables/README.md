# 05 -- extra LAMMPS observables

Pass scalars into LAMMPS, pull extra ones back out.

What this shows:
- `--invar <name> <value>` -- equivalent to adding `-var name value`
  on the LAMMPS command line. The script accesses it as `${<name>}`.
- `--outvar <name>` -- capture an additional LAMMPS variable per
  iter; it appears as its own column in `outvar.tadah` alongside the
  `--varloss` and `--hotvar` columns.

The companion script `../common/lammps/Ecoh_param.in` consumes
`${nx}`, `${ny}`, `${nz}` so the supercell size is set per-iter from
config.hpo rather than hard-coded in the LAMMPS file.

A PC_LAMMPS line takes any combination of `--invar`, `--outvar`,
`--hotvar`, `--varloss` and a handful of less-common flags (see the
HPO README for the full list).

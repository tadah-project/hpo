[![pipeline](https://git.ecdf.ed.ac.uk/tadah/hpo/badges/develop/pipeline.svg)](https://git.ecdf.ed.ac.uk/tadah/hpo/-/commits/develop)
[![coverage](https://git.ecdf.ed.ac.uk/tadah/hpo/badges/develop/coverage.svg)](https://git.ecdf.ed.ac.uk/tadah/hpo/-/commits/develop)
[![license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![version](https://img.shields.io/badge/version-1.3.0--beta.1-blue.svg)](https://git.ecdf.ed.ac.uk/tadah/hpo/-/tags)

# Tadah!HPO

Hyperparameter-optimisation engine of the
[Tadah!](https://tadah.readthedocs.io/) MLIP toolchain. Drives
parameter search around a Tadah! training+validation cycle, with
optional LAMMPS-driven physical-property targets (elastic constants,
equation-of-state points, surface energies, …).

The user-facing surface is one config file (`config.hpo`) read by
`tadah hpo`. Inside it you select:

- an **optimiser** (`DLIB` / `NLOPT` / `TADAH` algorithm),
- an **init strategy** (where to start the search),
- a **run policy** (single-shot, multistart, prestage, basins),
- **loss terms** (`PC_ERMSE`, `PC_FRMSE`, `PC_LAMMPS`, …),
- and an **`OPTIM` search space**.

For the full grammar, parallelism / performance notes, and worked
examples, see the
[HPO documentation](https://tadah.readthedocs.io/) and the
[`examples/`](examples/) directory.

## You don't need to use this directly

Tadah!HPO is an internal module of the Tadah! MLIP toolchain. It is
fetched automatically (via CMake `FetchContent`) by the umbrella
project [Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip) when
configured with `-DTADAH_ENABLE_NFP=ON`. End users install the
umbrella project; the `tadah hpo` CLI lives there.

## Contributing

Build instructions, dependencies, and layout live in
[CONTRIBUTING.md](CONTRIBUTING.md). The full multi-repo contributor
guide is on the umbrella project.

## License

GPL-3.0; see [LICENSE](LICENSE).

## Reporting issues

Bugs and feature requests:
[Tadah!HPO issues](https://git.ecdf.ed.ac.uk/tadah/hpo/-/issues).
For end-user questions please use the umbrella project
[Tadah!MLIP](https://git.ecdf.ed.ac.uk/tadah/tadah.mlip).

[![license](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![version](https://img.shields.io/badge/version-1.6.0-blue.svg)](https://github.com/tadah-project/hpo/tags)

# HPO

Hyperparameter-optimisation engine of the
[Tadah!](https://tadah.readthedocs.io/v1-dev/) MLIP toolchain. Drives
parameter search around a Tadah! training+validation cycle, with
optional LAMMPS-driven physical-property targets (elastic constants,
equation-of-state points, surface energies, …).

The user-facing surface is one config file (`config.hpo`) read by
`tadah hpo`. Inside it you select:

- an **optimiser** (`DLIB` / `NLOPT` / `TADAH` algorithm),
- an **init strategy** (where to start the search),
- a **run policy** (single-shot, multistart, prestage),
- **loss terms** (`PC_ERMSE`, `PC_FRMSE`, `PC_LAMMPS`, …),
- and an **`OPTIM` search space**.

For the full grammar and the parallelism / performance notes, see the
[HPO documentation](https://tadah.readthedocs.io/v1-dev/).

## You don't need to use this directly

HPO is an internal module of the Tadah! MLIP toolchain. It is
fetched automatically (via CMake `FetchContent`) by the umbrella
project [Tadah!MLIP](https://github.com/tadah-project/tadah.mlip) when
configured with `-DTADAH_ENABLE_NFP=ON`. End users install the
umbrella project; the `tadah hpo` CLI lives there.

## Contributing

Build instructions, dependencies, and layout live in
[CONTRIBUTING.md](CONTRIBUTING.md). The full multi-repo contributor
guide is on the umbrella project.

## License

GPL-3.0; see [LICENSE](LICENSE).

## Reporting issues

This repository is a component of Tadah! and has no issue tracker of
its own. Bugs, feature requests and questions go to the tracker of
the umbrella project,
[Tadah!MLIP](https://github.com/tadah-project/tadah.mlip/issues).

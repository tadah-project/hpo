# Contributing to HPO

The full multi-repo contributor guide for the Tadah! project lives on
the umbrella repository:
<https://git.ecdf.ed.ac.uk/tadah/tadah.mlip/-/blob/develop/CONTRIBUTING.md>

This file covers what is specific to HPO. Bug reports and
feature requests on this module use the GitLab issue tracker on this
project; cross-cutting changes (work that touches more than one
Tadah! module) should be discussed on the umbrella project first.

## Build & test in isolation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build/tests --output-on-failure
```

In-source builds are blocked deliberately; always pick a separate
build directory.

### Dependencies

- C++17 compiler.
- [CORE](https://git.ecdf.ed.ac.uk/tadah/core),
  [MODELS](https://git.ecdf.ed.ac.uk/tadah/models),
  [MLIP](https://git.ecdf.ed.ac.uk/tadah/mlip), and
  [MD](https://git.ecdf.ed.ac.uk/tadah/md) — fetched
  automatically.
- [dlib](http://dlib.net/), [NLopt](https://nlopt.readthedocs.io/),
  [Eigen3](https://eigen.tuxfamily.org/), and (optionally)
  [Ceres](http://ceres-solver.org/) — fetched automatically at pinned
  versions.
- LAPACK / BLAS at link time (provided by the build environment).
- A LAMMPS library built with [Tadah!LAMMPS](https://git.ecdf.ed.ac.uk/tadah/tadah.lammps)
  is needed at link time only when building HPO against the
  LAMMPS-driven targets (`PC_LAMMPS`). See the umbrella
  [installation guide](https://tadah.readthedocs.io/develop/installation.html#installation-nfp).

## Layout

| Path | Purpose |
|---|---|
| `include/tadah/hpo/` | Public headers (drivers, init strategies, run policies, loss terms). |
| `src/`               | Implementation translation units. |
| `tests/`             | Unit and integration tests. |
| `examples/`          | Worked-through `config.hpo` reference suite (see `run_reference_suite.sh`). |

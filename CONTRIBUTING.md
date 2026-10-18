# Contributing to HPO

The full multi-repo contributor guide for the Tadah! project lives on
the umbrella repository:
<https://git.ecdf.ed.ac.uk/tadah/tadah.mlip/-/blob/v1-dev/CONTRIBUTING.md>

This file covers what is specific to HPO. Bug reports and
feature requests on this module use the GitLab issue tracker on this
project; cross-cutting changes (work that touches more than one
Tadah! module) should be discussed on the umbrella project first.

## Build & test in isolation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
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
- Python 3.11 or later for the tests (`BUILD_TESTING`): the registry drift
  gate `hpo_anneal_registry_sync` reads TOML with `tomllib`, and the
  configure step refuses an older interpreter
  (`cmake/hpo_test_python.cmake`).
- A LAMMPS library built with [Tadah!LAMMPS](https://git.ecdf.ed.ac.uk/tadah/tadah.lammps)
  is needed at link time only when building HPO against the
  LAMMPS-driven targets (`PC_LAMMPS`). See the umbrella
  [installation guide](https://tadah.readthedocs.io/v1-dev/installation.html#installation-nfp).

## Layout

| Path | Purpose |
|---|---|
| `include/tadah/hpo/` | Public headers (drivers, init strategies, run policies, loss terms). |
| `src/`               | Implementation translation units. |
| `tests/`             | Unit and integration tests. |
| `examples/`          | Worked-through `config.hpo` reference suite (see `run_reference_suite.sh`). |
| `blueprint/`         | Registry files the documentation reads (see `blueprint/README.md`). |

## Generated files

Two checked-in files are generated, and a CTest entry fails when one
differs from what its source produces. Regenerate the file and commit it
with the change that alters it.

- `include/tadah/hpo/anneal_keys_gen.h`, from `blueprint/anneal_keys.toml`:

  ```bash
  python3 blueprint/gen_anneal_keys.py --header
  ```

  CTest `hpo_anneal_registry_sync`.
- `blueprint/optimizers_gen.toml`, the optimiser registry and the ALGO help
  of every optimiser, for the documentation (built without C++). Regenerate
  it after a change to `src/optimizer_caps.cpp`, the ALGO help
  (`src/optimizer_spec.cpp`, `src/asa_options.cpp`,
  `blueprint/anneal_keys.toml`) or the NLopt version:

  ```bash
  cmake --build build --target hpo_regenerate_optimizer_registry
  ```

  CTest `hpo_optimizer_registry_sync` (label `docs`).

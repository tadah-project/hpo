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
| `blueprint/`         | Registry files for the documentation (see `blueprint/README.md`). |

## Generated files

Some checked-in files are generated, and a CTest entry fails when one
differs from what produces it. Such a file is regenerated and committed
with the change that alters it.

- `include/tadah/hpo/anneal_keys_gen.h` (CTest `hpo_anneal_registry_sync`)
  and `blueprint/optimizers_gen.toml` (CTest
  `hpo_optimizer_registry_sync`): `blueprint/README.md` says what each is
  generated from and how to regenerate it.
- The `reference_fingerprint.json` of each example, which
  `examples/tools/generate_fingerprint.py` writes from a run of the example
  and its entry in `examples/tools/tolerance_overrides.json` (CTest
  `hpo_example_fingerprint_overrides_sync` fails when a fingerprint's
  fields from that file differ from it). After a change to the file, the
  example is run again (see `examples/README.md`) and its fingerprint is
  regenerated:

  ```bash
  python3 examples/tools/generate_fingerprint.py examples/<example> \
      --overrides examples/tools/tolerance_overrides.json
  ```

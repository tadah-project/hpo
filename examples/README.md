# HPO/examples

A graded sequence of runnable HPO examples. Each example is a
self-contained directory with one `config.hpo` and a `README.md` that
explains what the example exercises. All examples share the same tiny
Ta training+validation bundle in `common/`.

## How to run one

    cd 01_minimal
    bash ../common/run.sh

Override the binary path with `TADAH=/path/to/tadah` if `tadah` is not
on your PATH. Override OpenMP threads with `OMP_NUM_THREADS=N`. Each
run cleans its previous output (`best_*.tadah`, `loss.tadah`,
`outvar.tadah`, `params.tadah`, `stats.tadah`, `hpo_run.log`).

A handful of examples (09, 10) ship their own `run.sh` because they
need a `seed_pot.tadah` bootstrapped first; that script handles the
bootstrap on first invocation and reuses the seed thereafter.

## How to run them all

    bash run_all.sh

prints a one-line summary per example (best loss, eval count, status).

## How to check against the reference

    bash run_reference_suite.sh --full

re-runs every example and compares the output to a checked-in
`reference_fingerprint.json` (loss, hyperparameter values, eval count,
optimiser status). Use `--example=NN` to check a single example or
`--check-only` to skip the re-run.

## The sequence

The examples build up gradually: 01 is the smallest viable config,
each subsequent one introduces one or two new features.

| #  | Example                  | Introduces |
|----|--------------------------|------------|
| 01 | `01_minimal`             | Minimal `OPTIMIZER / STRATEGY / PC_LAMMPS / OPTIM` skeleton (DLIB:BFGS). |
| 02 | `02_random_search`       | `LIB TADAH ALGO RANDOM` + `SEED` (baseline sampler). |
| 03 | `03_nlopt_local`         | `LIB NLOPT ALGO LN_SBPLX` + `STEP` (derivative-free local). |
| 04 | `04_skip_bad_pots`       | `PC_<METRIC> --skip-above / --skip-rel` early-reject filter. |
| 05 | `05_extra_observables`   | `PC_LAMMPS --invar` / `--outvar` (extra LAMMPS observables). |
| 06 | `06_init_random`         | `INIT { STRATEGY RANDOM, SEED }`. |
| 07 | `07_init_lhs_multistart` | `INIT { STRATEGY LHS }` + `STRATEGY MULTISTART`. |
| 08 | `08_init_sobol_gfs`      | `INIT { STRATEGY SOBOL }` + `DLIB:GFS` multi-seed consumer. |
| 09 | `09_init_warm`           | `INIT { STRATEGY WARM <file> }`, warm-start cache. |
| 10 | `10_init_chain`          | Multi-strategy INIT chain. |
| 11 | `11_grid_sweep`          | `LIB TADAH ALGO GRID` exhaustive Cartesian sweep. |
| 12 | `12_prestage`            | `STRATEGY PRESTAGE { EXPLORE { ... } }` global -> refine. |
| 13 | `13_restart`             | `STRATEGY { ... RESTART { ... } }` decorator. |
| 14 | `14_mlsl_inner`          | `STRATEGY { ... INNER { ... } }` for AUGLAG / MLSL. |
| 15 | `15_simulated_annealing` | `LIB TADAH ALGO ANNEAL` with cooling-schedule PARAMs. |

## Files in `common/`

| File | Purpose |
|---|---|
| `config.tadah`            | Shared TADAH model (Blip 2B + mBlipEnv MB on Ta). |
| `config.val`              | Shared validation-set list. |
| `data/elastic.tadahdb`    | 29 elastic-deformation BCC structures. |
| `data/eos_bcc.tadahdb`    | EOS sweep over the BCC volume. |
| `lammps/Ecoh.in`          | Cohesive-energy LAMMPS script (4x4x4 BCC). |
| `lammps/Ecoh_param.in`    | Same, but the supercell is `${nx} x ${ny} x ${nz}`. |
| `run.sh`                  | Driver script: clean, invoke `tadah hpo`, summarise. |

All paths in `config.tadah` are relative to the example directory; that
is why `run.sh` `cd`s into the example before invoking `tadah`.

## Notes on coverage

These examples exercise every `STRATEGY` and `INIT` flavour and the
`LIB` families that today work with `PC_LAMMPS`. `STRATEGY BASINS` and
`STRATEGY BASINSV2` are intentionally not shown -- both are still
experimental. `CERES:LBFGS/WOLFE/NLCG/STEEPEST_DESCENT` are also not
shown here: they are gradient-based and currently incompatible with
`PC_LAMMPS` (use `LIB DLIB ALGO BFGS` or a gradient-free family for
LAMMPS-driven loss terms).

## Tightening `--hotvar` against unphysical lattices

A reasonable `--hotvar` window on hot PC_LAMMPS lines (e.g.
`--hotvar hot_a0 2.5 4.5 3.302` for Ta) is the cheapest extra filter
against runaway HPO candidates: an out-of-range value fail-scores the
iter and skips every cold script.

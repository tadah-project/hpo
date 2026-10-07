# HPO/examples

A graded sequence of runnable HPO examples. Each example is a
self-contained directory with one `config.hpo` and a `README.md` that
explains what the example exercises. All examples share the same tiny
Ta training+validation bundle in `common/`.

## How to run one

    cd 01_minimal
    bash ../common/run.sh

Override the binary path with `TADAH=/path/to/tadah` if `tadah` is not
on your PATH. `THREADS` in each config sets the main shared worker-pool
width; `OMP_NUM_THREADS` controls only residual OpenMP kernels and does
not multiply it. Each run cleans its previous output (`best_*.tadah`,
`loss.tadah`, `outvar.tadah`, `params.tadah`, `stats.tadah`,
`hpo_run.log`).

`common/run.sh` passes the example's own `config.tadah` and `config.val`
when the example directory has them (22 and 23 do), else the shared ones
in `common/`, and prints which files it uses.

Five examples ship their own `run.sh`: 09 and 10 bootstrap a
`seed_pot.tadah` on first invocation (and reuse it thereafter), 16 runs
with its local `config.tadah`, 17 trains the shared model and pins its
weights before the HPO run, and 27 checks the EMBF0 ZERO contract on its
outputs. Run those with `bash ./run.sh`.

## How to run them all

    bash run_all.sh

prints a one-line summary per example (best loss, eval count, status).

## How to check against the reference

    bash run_reference_suite.sh --full

re-runs every example and compares the output to a checked-in
`reference_fingerprint.json` (loss, hyperparameter values, eval count,
optimiser status, the `NaN/Inf clampings` and `Evaluations that threw`
counts of `hpo_run.log`, which must not exceed the fingerprint's
`nan_clampings_max` and `threw_max`, 0 unless
`tools/tolerance_overrides.json` allows more, and for 25 and 26 lines
that `hpo_run.log` must contain: 25's six band-A graded failures, 26's
worker processes and hard deadline). Use `--example=NN` to check a single example or
`--check-only` to skip the re-run. With `TADAH_PC_ISOLATION=THREAD` set
(for environments that block the worker socket) 26's journal checks are
not applied and the suite prints a `NOTE` line for each, so PROCESS mode
is only checked by a run without that override.

## The sequence

The examples build up gradually: 01 is the smallest viable config,
each subsequent one introduces one or two new features.

| #  | Example                  | Introduces |
|----|--------------------------|------------|
| 01 | `01_minimal`             | Minimal `OPTIMIZER / STRATEGY / PC_LAMMPS / OPTIM` skeleton (DLIB:BFGS). |
| 02 | `02_random_search`       | `LIB TADAH ALGO RANDOM` + `SEED` (baseline sampler). |
| 03 | `03_nlopt_local`         | `LIB NLOPT ALGO LN_SBPLX` + `STEP` (fraction of each OPTIM width; derivative-free local). |
| 04 | `04_skip_bad_pots`       | `PC_<METRIC> --skip-above / --skip-rel` early-reject filter. |
| 05 | `05_extra_observables`   | `PC_LAMMPS --invar` / `--outvar` (extra LAMMPS observables). |
| 06 | `06_init_random`         | `INIT { METHOD RANDOM, SEED }`. |
| 07 | `07_init_lhs_multistart` | `INIT { METHOD LHS }` + `STRATEGY { TYPE MULTISTART }`. |
| 08 | `08_init_sobol_gfs`      | `INIT { METHOD SOBOL }` + `DLIB:GFS` multi-seed consumer. |
| 09 | `09_init_warm`           | `INIT { METHOD WARM <file> }`, warm-start cache. |
| 10 | `10_init_chain`          | Multi-strategy INIT chain. |
| 11 | `11_grid_sweep`          | `LIB TADAH ALGO GRID` exhaustive Cartesian sweep. |
| 12 | `12_prestage`            | `STRATEGY { TYPE PRESTAGE EXPLORE { ... } }` global -> refine. |
| 13 | `13_restart`             | `STRATEGY { ... RESTART { ... } }` decorator. |
| 14 | `14_mlsl_inner`          | `STRATEGY { ... INNER { ... } }` for AUGLAG / MLSL. |
| 15 | `15_simulated_annealing` | `LIB TADAH ALGO ANNEAL` with cooling-schedule PARAMs. |
| 16 | `16_fixweight_mixed`     | `FIXINDEX`/`FIXWEIGHT` pinning + `OPTIM FIXWEIGHT` (mixed LS/pinned/searched weights; local `config.tadah`). |
| 17 | `17_no_training_db`      | No training `DBFILE`, no `--validation`: all coefficients pinned, `PC_LAMMPS`-only loss (refit-style polish). |
| 18 | `18_anneal_pt`            | `LIB TADAH ALGO ANNEAL_PT` parallel tempering with replica exchanges and cumulative/windowed swap-flow diagnostics. |
| 19 | `19_anneal_pa`            | `LIB TADAH ALGO ANNEAL_PA` population annealing with ESS-controlled resampling and explicit all-failed handling. |
| 20 | `20_anneal_corana`        | `LIB TADAH ALGO ANNEAL` with adaptive Corana coordinate steps and boundary-policy diagnostics. |
| 21 | `21_asa`                  | `LIB ASA ALGO ASA` — vendored Ingber ASA-30.53 (adaptive simulated annealing) with original option spellings (`PARAM Temperature_Ratio_Scale` etc.). |
| 22 | `22_pc_observable`        | `PC_OBSERVABLE cohesive_energy` in place of `PC_LAMMPS` (no LAMMPS); an unset `GRAD_STRATEGY` resolves to `ANALYTICAL` for `DLIB:BFGS` (local `config.tadah`: `OALGO 4`, `LAMBDA`). |
| 23 | `23_per_dataset_targets`  | `PC_ERMSE (1)` / `PC_ERMSE (2)`: a target and weight per validation dataset, per-dataset `--skip-above` (local `config.val`). |
| 24 | `24_optim_auto`           | `OPTIM SGRIDMB (1) AUTO`: a seed-relative box; resolves to 03's `[0.1, 0.4]` and reproduces 03. |
| 25 | `25_failprobe_grading`    | `FAILSCORE_GRADING ON`, `PC_FAILPROBE` (`common/lammps/failprobe.in`); a lattice window makes graded (band A) failures. |
| 26 | `26_pc_isolation`         | `PC_ISOLATION PROCESS` + `PC_WORKER_TIMEOUT`; same results as 01. |
| 27 | `27_dm_ream_embf0`        | DM_REAM `tadah refit` seed (`RHOFORM SIGNED`) with the zero-referenced embedding (`EMBF0 ZERO`): `OPTIM CEMBFUNC` on the negative embedding knots; `run.sh` asserts the isolated-atom energy is exactly 0 for every candidate. |

## Files in `common/`

| File | Purpose |
|---|---|
| `config.tadah`            | Shared TADAH model (Blip 2B + mBlipEnv MB on Ta). |
| `config.val`              | Shared validation-set list. |
| `data/elastic.tadahdb`    | 29 elastic-deformation BCC structures. |
| `data/eos_bcc.tadahdb`    | EOS sweep over the BCC volume. |
| `lammps/Ecoh.in`          | Cohesive-energy LAMMPS script (4x4x4 BCC). |
| `lammps/Ecoh_param.in`    | Same, but the supercell is `${nx} x ${ny} x ${nz}`. |
| `lammps/failprobe.in`     | Minimiser-free sanity probe `S` for `PC_FAILPROBE` (the documentation's example). |
| `run.sh`                  | Driver script: clean, invoke `tadah hpo` (with the example's own `config.tadah` / `config.val` when present), summarise. |

All paths in `config.tadah` are relative to the example directory; that
is why `run.sh` `cd`s into the example before invoking `tadah`.

## Notes on coverage

These examples exercise the documented `STRATEGY` types (`PLAIN`,
`MULTISTART`, `PRESTAGE`, the `RESTART` decorator), every `INIT` method
and the `LIB` families that today work with `PC_LAMMPS`.
`CERES:LBFGS/WOLFE/NLCG/STEEPEST_DESCENT` and the gradient-based NLopt
algorithms (`LD_*`, `GD_*`) are not shown here: `PC_LAMMPS` rejects
every gradient-based backend except `DLIB:BFGS` (use it or a
gradient-free family for LAMMPS-driven loss terms). They do run with
`PC_OBSERVABLE` and validation metrics, where example 22 shows the
analytical gradient with `DLIB:BFGS`.

`tests/test_examples_parse.cpp` (a CTest case per example) parses every
`config.hpo` and applies the startup checks. A new example needs a line
in `tools/parse_guard.manifest`, which also lists the files an example's
`run.sh` creates before the run (e.g. 09's `seed_pot.tadah`).

The INIT examples write chain entries as `STRATEGY <method>`, the alias
of the canonical `METHOD <method>`; both spellings parse identically.

## Tightening `--hotvar` against unphysical lattices

A reasonable `--hotvar` window on hot PC_LAMMPS lines (e.g.
`--hotvar hot_a0 2.5 4.5 3.302` for Ta) is the cheapest extra filter
against runaway HPO candidates: an out-of-range value fail-scores the
iter and skips every cold script.

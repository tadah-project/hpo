# HPO/examples

Fifteen runnable end-to-end examples covering every block in the new
`STRATEGY`-rooted grammar — `INIT`, `NOISE`, `RESTART`, `EXPLORE`,
`REFINE`, `INNER`. Each example directory is self-contained: one
`config.hpo`, one `README.md` describing what the example shows, and a
`run.sh` (for examples that need bootstrapping). They share the small
training/validation bundle in `common/`.

## Grammar at a glance

```
OPTIMIZER
  INIT          # init chain (STRATEGY/K_TOTAL/SEED/INCLUDE_CONFIG/WARM_ON_MISMATCH)
    ...
  ENDINIT
  NOISE         # NOISE.AUTO + NOISE.CALIBRATION
    ...
  ENDNOISE
  STRATEGY      # mandatory; TYPE picks the policy
    TYPE PLAIN | BASINS | MULTISTART | PRESTAGE
    LIB ...   ALGO ...        # the (default) inner driver
    ...stop / param keys...
    INNER     # MLSL/AUGLAG-family inner local sub-optimiser
    ENDINNER
    EXPLORE   # TYPE BASINS / TYPE PRESTAGE — global stage
    ENDEXPLORE
    REFINE    # TYPE BASINS — per-basin local stage
    ENDREFINE
    RESTART   # decorator (PLAIN / MULTISTART / PRESTAGE)
    ENDRESTART
  ENDSTRATEGY
ENDOPTIMIZER
```

## How to run one example

```
cd 01_baseline_bfgs
bash ../common/run.sh
```

Each run cleans its previous output, invokes `tadah hpo` from the
example dir, and prints a one-line summary. Outputs (`best_*.tadah`,
`loss.tadah`, `hpo_run.log`, ...) land in the example dir.

## How to run them all

```
bash run_all.sh
```

Prints a summary table; ~30 seconds total on a quiet machine.

## Files in `common/`

| File | Purpose |
|---|---|
| `data/elastic.tadahdb` | 29 elastic-deformation BCC structures |
| `data/eos_bcc.tadahdb` | EOS sweep over the BCC volume |
| `lammps/Ecoh.in` | Cohesive-energy LAMMPS script (uses `${hot_a0}` warm-start) |
| `config.tadah` | Shared TADAH model (Blip 2B + mBlipEnv MB) |
| `config.val` | Shared validation-set list |
| `run.sh` | Driver script — clean + invoke `tadah hpo` + summarise |

The training and validation lists overlap on `elastic.tadahdb`
(intentionally — keeps the example tiny). All paths in `config.tadah`
are relative to the example directory; that is why `run.sh` `cd`s into
the example before invoking tadah.

## What is anonymised

The `data/*.tadahdb` files are slimmed copies of public DFT data. The
first column of each structure entry (originally an absolute filesystem
path on the author's machine) was rewritten to `examples/data/...` so
no private path leaks. Element identity (Ta) is retained — it is a
public chemical symbol.

## Index

| # | Example | Knobs exercised |
|---|---|---|
| 01 | `01_baseline_bfgs` | `STRATEGY { TYPE PLAIN }` — implicit CONFIG, single-shot DLIB:BFGS |
| 02 | `02_random_seeded` | `INIT { STRATEGY RANDOM, INCLUDE_CONFIG false, SEED }` |
| 03 | `03_lhs_multistart` | `INIT { STRATEGY LHS }` + `STRATEGY { TYPE MULTISTART, N_STARTS, BUDGET_PER_START }` |
| 04 | `04_sobol_seeded_gfs` | `INIT { STRATEGY SOBOL }`; DLIB:GFS consumes all K seeds |
| 05 | `05_warm_refinement` | `INIT { STRATEGY WARM, WARM_ON_MISMATCH }`, HOT_A0 cache |
| 06 | `06_prior_seeded` | `INIT { STRATEGY PRIOR }`, descriptor `suggested_value` |
| 07 | `07_stage_global` | `STRATEGY { TYPE PRESTAGE, TOPK, EXPLORE { ... } }` |
| 08 | `08_restart_on_stagnation` | `STRATEGY { RESTART { AFTER_STAGNATION, MAX, STRATEGY } }` |
| 09 | `09_noise_auto_bfgs` | `NOISE { AUTO }` (bareword form) |
| 10 | `10_derivative_free_local` | `STRATEGY { TYPE PLAIN, LIB NLOPT, ALGO LN_SBPLX, STEP }` |
| 11 | `11_chained_init_warm_lhs` | Chained init: WARM + LHS + DLIB:GFS, WARM-front reorder |
| 12 | `12_random_search` | `STRATEGY { TYPE PLAIN, LIB TADAH, ALGO RANDOM, SEED }` |
| 13 | `13_nlopt_crs_global` | `LIB NLOPT, ALGO GN_CRS2_LM` — pure stochastic global, no internal local refinement |
| 14 | `14_nlopt_mlsl_lds` | `LIB NLOPT, ALGO G_MLSL_LDS` + `INNER { LIB NLOPT, ALGO LN_BOBYQA }` — MLSL global+local hybrid |
| 15 | `15_basins_run_and_forget` | `STRATEGY { TYPE BASINS, EXPLORE { FTOL_LADDER, BUDGET_* }, REFINE { ... } }` |
| 16 | `16_simulated_annealing` | `STRATEGY { TYPE PLAIN, LIB TADAH, ALGO ANNEAL, PARAM RT/L/STEP_FRAC/T0_PROBES/T0_PACCEPT }` — classic Kirkpatrick/Metropolis SA with auto-T0 calibration |

## Reading the diagnostics

Each example's `hpo_run.log` records the resolved configuration at the
top, including a new `Init strategy:` and `Run policy:` block:

```
  Init strategy:
    [1] SOBOL 4
    K_TOTAL            : 4
    SEED               : 31337
    INCLUDE_CONFIG     : false
    WARM_ON_MISMATCH   : warn
  Run policy:
    (TYPE PLAIN — single-shot dispatch)
```

This block is the one place to confirm which features actually fired.
On stdout, lines tagged `[HPO] ...` mark each phase: init resolution,
seed forwarding, multistart starts, stagnation restarts, noise-auto
fd_step, etc.

## Beware (limitations)

- `PARALLEL > 1` is parsed but starts run sequentially —
  HPO logs a warning if you set it >1.
- `DLIB:GFS` consumes every seed from the chain (patched). Other
  optimisers consume only `optim_init.front()`; HPO logs a warning
  when the chain produced more seeds than the optimiser can use.
- Tightening `TADAH_HPO_A0_MIN`/`TADAH_HPO_A0_MAX` (e.g. `2.5/4.5`
  for Ta) is the most effective extra filter against runaway HPO
  candidates that produce a non-physical lattice.

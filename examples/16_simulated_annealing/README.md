# 16 — TADAH:ANNEAL classic Simulated Annealing

Kirkpatrick/Metropolis Simulated Annealing in the in-tree `TADAH`
optimiser library. Operates in transformed (post-`LOG_HP`) parameter
space; the LAMMPS side always sees real-space values.

References:

1. Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. *Optimization by
   simulated annealing.* Science 220 (4598), 671–680, 1983.
2. Metropolis, N., Rosenbluth, A., Rosenbluth, M., Teller, A., and
   Teller, E. *Equation of State Calculations by Fast Computing
   Machines.* J. Chem. Phys. 21, 1087–1092, 1953.

## What is going on

Each temperature rung runs `L` Metropolis trials: propose
`x' = x + sigma * N(0,I)`, accept if `Δf <= 0` or with probability
`exp(-Δf / T)`. Temperature decays geometrically: `T <- T * RT`.

`T_INIT` is auto-calibrated from `T0_PROBES` uniform-bounds samples
so the early acceptance ratio is ≈ `T0_PACCEPT` (default 0.8).
Override by setting `PARAM T_INIT <value>` — that skips calibration
and spends those probe evals on the SA loop instead.

## Knobs

| PARAM         | Default      | Effect |
|---------------|--------------|--------|
| `T_INIT`      | auto         | Initial temperature. Skips calibration when set. |
| `T_MIN`       | `1e-12`      | Cooling floor; loop stops when `T < T_MIN`. |
| `RT`          | `0.85`       | Geometric cooling factor in `(0,1)`. |
| `L`           | `50`         | Markov-chain length per temperature. |
| `STEP_FRAC`   | `0.10`       | `sigma_i = STEP_FRAC * (high_i - low_i)` per dim. |
| `T0_PROBES`   | `20`         | Random probes used to calibrate T0. |
| `T0_PACCEPT`  | `0.80`       | Target acceptance probability for T0. |

`STEP <s>` (top-level) overrides `STEP_FRAC` with a uniform sigma.
`SEED <int>` seeds the RNG.

## Stops honoured

- `MAXEVAL`: hard cap on `host()` calls **including** T0 probes.
- `MAXTIME`: wall-clock cap.
- `STOPVAL`: stop on first eval `<= STOPVAL`.
- `T_MIN`: SA-specific cooling floor.

`FTOL_REL` / `FTOL_ABS` are **not** honoured by this driver.

## Diagnostic lines

The two stdout lines that mark the run starting:

```
[HPO] ANNEAL: seed=1234  L=20  RT=0.85  T_MIN=1e-12  STEP_FRAC=0.15
[HPO] ANNEAL: T0=8.188e+00 from 20 probes,  mean|dE|=1.827e+00  loss range=[2.27e-01, 5.50e+00]  std=1.58e+00  target p_accept=8.000e-01
```

`hpo_run.log` additionally records, for each cooling rung:

```
ANNEAL rung k=0  T=8.188e+00  proposed=20  accepted=14 (70.0%)  oob=2 (9.1%)  f_x=2.31e-01  best=1.84e-01
ANNEAL rung k=1  T=6.960e+00  proposed=20  accepted=11 (55.0%)  oob=0 (0.0%)  f_x=1.97e-01  best=1.84e-01
...
```

and a final summary block:

```
ANNEAL summary: rungs=12  proposed=240  accepted=98 (40.8%)  oob=4 (1.6%)  T0=8.188e+00  T_final=2.46e-01  best=8.71e-02
ANNEAL accept-rate trajectory: min=20.0%  max=70.0%
ANNEAL stopped: MAXEVAL exhausted after 260 host eval(s) (12 rung(s) completed)
```

These tell you whether T0/RT were sensible (accept rate trajectory),
whether `STEP_FRAC` was too large (`oob` fraction), and which
condition tripped the loop.

If T0 calibration cannot make headway (`mean|dE|=0`, e.g. saturated
`FAILSCORE` everywhere), HPO falls back to `T0=1.0` and emits a
`WARNING` line — pick `PARAM T_INIT` or widen the bounds in that case.

When the top-level `STEP <s>` is set, the entry banner reports
`sigma=<s> (uniform; STEP override)` instead of `STEP_FRAC=<frac>`.

## Why it is useful

SA shines on rough, multi-modal landscapes where pure local methods
(`DLIB:BFGS`, `NLOPT:LN_BOBYQA`) get stuck and where pure random
search (example 12) wastes evals far from good basins. The
Metropolis acceptance gives the chain a probabilistic exit from
local minima that a strict descent cannot escape.

# 01 — single-shot DLIB:BFGS baseline

The simplest HPO run: `STRATEGY { TYPE PLAIN }` with no `INIT` chain
and no decorators. Iter 0 is the implicit CONFIG point.

## What you should observe

- `[HPO] InitStrategy CONFIG produced 1 starting point` — iter 0 is the
  in-config value next to the OPTIM line (here `RCUT2B = 5.6`).
- `[HPO] DLIB:BFGS dim=1` — the dispatcher confirms the algorithm.
- `hpo_run.log` records `Init strategy: (default — implicit CONFIG only)`
  and `Run policy: (TYPE PLAIN — single-shot dispatch)`.

## Run

```
cd 01_baseline_bfgs
bash ../common/run.sh
```

## Why use it

Use as a baseline when comparing more elaborate configurations. Knowing
the loss this minimal recipe reaches gives you a reference for whether
the extra exploration in `INIT { STRATEGY SOBOL ... }`, `STRATEGY {
TYPE MULTISTART ... }`, or `STRATEGY { TYPE PRESTAGE ... }` is paying
for itself.

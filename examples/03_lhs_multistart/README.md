# 03 — LHS seeds, MULTISTART × BFGS

Spreads four Latin-hypercube seeds across the search box and runs BFGS
from each. The host tracks the global best across all four trajectories.

## Knobs introduced

- `INIT { STRATEGY LHS K }` — Latin-hypercube: one sample per stratum
  per dim. Fills small `K` more evenly than RANDOM at the same `K`.
- `STRATEGY { TYPE MULTISTART, N_STARTS K }` — run the inner optimiser
  `K` independent times. Each start consumes one seed from the resolved
  init chain.
- `STRATEGY { BUDGET_PER_START N }` — per-start MAXEVAL override. When
  unset, the outer MAXEVAL is split equally (ceil division), so the
  total budget remains bounded.

## What to look for

- `[HPO] InitStrategy LHS produced 4 starting points` — chain produced
  enough seeds for all starts.
- `[HPO] MULTISTART per-start MAXEVAL = 3 (from BUDGET_PER_START)`
  — the budget knob is honoured.
- Four `[HPO] MULTISTART start i/4 — running inner optimiser` lines
  bracket each per-start dispatch.
- `hpo_run.log → Run policy: MULTISTART : N_STARTS=4 (PARALLEL=1) BUDGET_PER_START : 3`
  records the configuration.

## Compatibility

- `STRATEGY { TYPE MULTISTART, N_STARTS > 1 }` is rejected at
  validate-time for `NLOPT:GN_*` algorithms (they consume only one
  starting point). Use `LD_*` / `LN_*` or `DLIB:GFS` / `DLIB:BFGS`
  instead.
- For `DLIB:GFS`, all `K` seeds are forwarded into
  `find_min_global`'s `initial_function_evals` and the global engine
  runs once — see example 04.

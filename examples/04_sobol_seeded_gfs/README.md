# 04 — SOBOL seeds, DLIB:GFS

Quasi-random space-filling start fed into dlib's global optimiser. GFS
treats seeds as already-known function evaluations, so its UCB stop
criterion tightens from iter 0.

## Knobs introduced

- `INIT { STRATEGY SOBOL K }` — `K` deterministic low-discrepancy
  points in the (transformed) box. With the same `INIT { SEED }`, the
  same `K` points come out.
- `STRATEGY { TYPE PLAIN, LIB DLIB, ALGO GFS }` — global optimiser with
  bounded sampling. Unlike BFGS it consumes every seed from the chain
  (not just the first).

## What to look for

- `[HPO] InitStrategy SOBOL produced 4 starting points`.
- `[HPO] DLIB:GFS dim=2`.
- `[HPO] DLIB:GFS seeding with 4 starting points` — confirms all four
  seeds are forwarded into `find_min_global` as
  `initial_function_evals`. This is the patched behaviour: previously
  only the first seed was used.
- `MAXEVAL 8` includes the seed evals (each seed counts as one), so
  GFS gets `8 - 4 = 4` extra acquisition steps.

## Notes

- `DLIB:GFS` does NOT need `STRATEGY { TYPE MULTISTART }` — it consumes
  the K seeds itself in a single call.
- Sobol determinism via `INIT { SEED }` is implemented as a discard
  count on the underlying boost::random::sobol engine; different seeds
  pick different sub-sequences but the points remain quasi-random.

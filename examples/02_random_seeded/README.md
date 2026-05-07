# 02 — RANDOM seed feeds DLIB:BFGS

A user with a poorly-trusted in-config value. They drop CONFIG via
`INIT { INCLUDE_CONFIG false }`, ask for one uniform-random seed inside
the bounds, then refine with BFGS.

## Knobs introduced

- `INIT { STRATEGY RANDOM K }` — `K` uniform samples in (transformed) bounds.
- `INIT { INCLUDE_CONFIG false }` — drop the implicit CONFIG; iter 0 is
  the RANDOM point exclusively.
- `INIT { SEED N }` — make the RANDOM draw reproducible.

## What to look for

- `[HPO] InitStrategy RANDOM produced 1 starting point` — confirms the
  RANDOM seed was used (not CONFIG).
- `Init strategy: [1] RANDOM 1 / INCLUDE_CONFIG: false` in
  `hpo_run.log` records both the explicit chain and the suppression.
- The first eval's loss is generally worse than CONFIG's because we are
  starting from a random point inside the box; BFGS subsequent evals
  walk back to a similar minimum.

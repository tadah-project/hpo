# 11 — Chained init: WARM + LHS feeding DLIB:GFS

Demonstrates that init strategies compose. Start from a WARM seed for
the local basin you trust, AND scatter a few LHS points to give the
global engine variety. DLIB:GFS consumes all of them as
`initial_function_evals`.

## Pattern

```
INIT
  STRATEGY LHS 3           # user wrote LHS first...
  STRATEGY WARM seed_pot.tadah   # ...and WARM second
ENDINIT
```

`build_init_strategy()` reorders this so iter 0 is the warm point and
the LHS draws follow. The reorder fires the diagnostic
`[HPO] InitStrategy: WARM moved to chain front`.

## What to look for

- The reorder diagnostic above.
- `[HPO] InitStrategy CHAIN produced 4 starting points` (1 WARM + 3 LHS).
- `[HPO] DLIB:GFS seeding with 4 starting points`.
- The first eval is the warm point; evals 2-4 are the LHS scatter; the
  rest are GFS acquisition steps.

## Run

The example's `run.sh` creates `seed_pot.tadah` from example 01 if
needed — same bootstrap pattern as example 05.

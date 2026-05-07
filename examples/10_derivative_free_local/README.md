# 10 — Derivative-free local: NLOPT:LN_SBPLX

LAMMPS-driven evaluations are noisy. `LN_SBPLX` (Subplex) is a
derivative-free local method that does not differentiate the loss, so
it is robust to small-amplitude noise.

## Knobs introduced

- `STRATEGY { LIB NLOPT, ALGO LN_SBPLX }` — derivative-free local
  optimiser. Other good choices in this family: `LN_BOBYQA`,
  `LN_NELDERMEAD`, `LN_COBYLA`.
- `STRATEGY { STEP 0.05 }` — initial step in the (transformed) space.
  Too small → SBPLX shrinks below precision; too large → wasted
  exploration.

## What to look for

- `[HPO] NLopt finished: SUCCESS  fmin=...` on success.
- For BOBYQA: a startup warning if the seed is within `STEP` of a
  bound (BOBYQA silently snaps the start to the interior). HPO
  auto-shrinks the step when possible. SBPLX has the analogous issue
  surfaced via NLopt's internal `errmsg` channel; HPO catches that and
  reports the masked status.

## Notes

- `STRATEGY { TYPE MULTISTART, N_STARTS > 1 }` is *rejected at
  validate-time* for `NLOPT:GN_*` — those globals consume one starting
  point only. `LN_*` and `LD_*` accept MULTISTART.
- NLopt drives its own log finish (HPO checks
  `RunPolicy::driver_finishes_log`), so the per-eval flush at the end
  of an NLopt run goes through a slightly different path.

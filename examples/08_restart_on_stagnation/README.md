# 08 — RESTART decorator (stagnation watchdog)

Long, unattended runs sometimes lock onto a local minimum that the
local engine cannot escape. RestartPolicy notices when no new best has
landed for `AFTER_STAGNATION` evals, draws a fresh init point,
and re-enters the inner optimiser.

## Knobs introduced

- `STRATEGY { RESTART { ... } ENDRESTART }` — decorator block.
  Composes with `TYPE PLAIN`, `TYPE MULTISTART`, and `TYPE PRESTAGE`.
- `RESTART { AFTER_STAGNATION N }` — evaluations without a new best
  before triggering a restart. `0` disables (default).
- `RESTART { MAX M }` — cap on how many restarts. After the cap is hit
  the run completes naturally.
- `RESTART { STRATEGY <method> [args] }` — the init chain used to draw
  the fresh starting point. Implicit CONFIG is *not* prepended here
  (the previous cycle already had CONFIG) — write the strategy you
  want explicitly. Same grammar as the OUTER `INIT` block.

## What to look for

- After a stretch of evals without improvement, the per-eval line
  emits and HPO catches `StagnationStop`:
  `[HPO] RESTART 1/2 — fresh init from RANDOM`
- `RESTART: stagnation reached, max restarts (2) used; stopping` is
  the final message when `MAX` is reached.
- The `Run policy:` block in `hpo_run.log` records the watchdog
  configuration verbatim.

## Notes

- Each fresh start gets its own stagnation window; counts do not
  accumulate across restarts.
- RESTART does NOT compose with `STRATEGY { TYPE BASINS }` — BASINS
  drives its own global+local cycle and supersedes the decorator stack.

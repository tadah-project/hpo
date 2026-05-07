# 07 — Staged global: GFS EXPLORE → BFGS refinement

`STRATEGY { TYPE PRESTAGE }` wraps a global-exploration optimizer in
the `EXPLORE { ... } ENDEXPLORE` sub-block. The EXPLORE phase runs
first; while it runs HPO records the top-K best (loss, params) pairs.
After it ends, the STRATEGY block's outer `LIB`/`ALGO` runs as a
MULTISTART over the K seeds (the **refinement** phase).

## Knobs introduced

- `STRATEGY { TYPE PRESTAGE }` — selects the staged-global policy.
- `EXPLORE { ... } ENDEXPLORE` — self-contained sub-spec for the global
  exploration phase (LIB, ALGO, MAXEVAL, INIT, ...).
- `TOPK K` — number of seeds carried from EXPLORE to refinement.
- The EXPLORE block runs FIRST. The outer `LIB`/`ALGO` runs
  second on each top-K seed.

## What to look for

- `[HPO] PRESTAGE EXPLORE: DLIB:GFS (MAXEVAL=6)` — confirms which
  engine handles the exploration.
- After EXPLORE finishes:
  `[HPO] PRESTAGE: top-2 losses: X.XXX Y.YYY` shows snapshot quality.
- `[HPO] REFINEMENT: refining top-2 seed(s) via DLIB:BFGS` — confirms
  refinement is starting.
- The outer `MAXEVAL 4` is the per-seed MAXEVAL of refinement (not
  shared) when refinement runs as MULTISTART over the seeds.

## Compatibility

- `EXPLORE` and `INNER` are independent: `INNER { ... } ENDINNER`
  carries NLopt's MLSL/AUGLAG inner sub-algorithm. You can nest INNER
  inside EXPLORE for `LIB NLOPT, ALGO G_MLSL_LDS` exploration.
- The legacy `STAGE_GLOBAL true` keyword has been removed; HPO
  parse-errors with a message pointing to `STRATEGY { TYPE PRESTAGE }`.

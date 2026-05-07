# 14 — NLopt MLSL_LDS + LN_BOBYQA (global+local hybrid)

Multi-Level Single-Linkage on a Low-Discrepancy Sequence. MLSL
samples the box on an LDS and launches a local descent from each
sample, skipping starts that fall in an already-discovered basin.
The result: a curated list of distinct local minima.

## Knobs introduced

- `STRATEGY { LIB NLOPT, ALGO G_MLSL_LDS }` — MLSL with LDS sampling
  (preferred over `G_MLSL` which uses uniform random sampling).
- `STRATEGY { POPULATION <N> }` — number of LDS sample points (each
  spawns a local descent).
- `INNER { ... } ENDINNER` — **required** for MLSL: specifies the
  per-sample local algorithm (here `LN_BOBYQA`). Renamed from the old
  `LOCAL/ENDLOCAL` block to remove the clash with BASINS' REFINE stage.

## What to look for

- `[HPO] NLOPT:G_MLSL_LDS   dim=...` — confirms which optimiser ran.
- BFGS-style or BOBYQA-style local descents nested inside the global
  loop — visible as bursts of `[HPO] BEST` events, separated by
  jumps to a fresh LDS sample point.
- The journal shows the INNER sub-spec under the OPTIMISER section
  (`Inner optimizer` / `Inner stop criteria`).

## Two patterns built on MLSL

### a) Single-call (this example) — let MLSL decide budget split.

### b) Staged — MLSL as the EXPLORE phase, refine top-K externally.

```
OPTIMIZER
  STRATEGY
    TYPE PRESTAGE
    LIB NLOPT
    ALGO LN_BOBYQA
    MAXEVAL 80
    TOPK 4
    EXPLORE
      LIB NLOPT
      ALGO G_MLSL_LDS
      MAXEVAL 600
      POPULATION 32
      INNER
        LIB NLOPT
        ALGO LN_BOBYQA
      ENDINNER
    ENDEXPLORE
  ENDSTRATEGY
ENDOPTIMIZER
```

The top-K snapshot then exposes basin diversity in the journal —
useful when you want to inspect what MLSL found before paying for
refinement.

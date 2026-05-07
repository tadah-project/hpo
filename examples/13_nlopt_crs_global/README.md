# 13 — NLopt CRS (pure stochastic global)

`GN_CRS2_LM` is Controlled Random Search with local mutation. Unlike
`DLIB:GFS`, it has no internal local-refinement phase: the algorithm
spends every evaluation on stochastic exploration of the box.

## Knobs introduced

- `STRATEGY { TYPE PLAIN, LIB NLOPT, ALGO GN_CRS2_LM }` — the algorithm.
- `INIT { SEED <N> }` — deterministic CRS population initialisation.

## What to look for

- `[HPO] NLOPT:GN_CRS2_LM   dim=...` — confirms which optimiser ran.
- The `loss.tadah` trace should *not* converge to a single value
  early — CRS keeps exploring until MAXEVAL.
- HPO may emit a warning that the chain produced more seeds than
  CRS will use (NLopt globals consume a single starting point only,
  the rest of the population is generated internally).

## When to use

- As a self-contained global pre-stage, then follow up with a
  separate WARM-refined local run.
- As the EXPLORE block of a `STRATEGY { TYPE PRESTAGE }` config (mirror
  example 07's pattern but with `LIB NLOPT, ALGO GN_CRS2_LM` instead of
  `DLIB:GFS`).

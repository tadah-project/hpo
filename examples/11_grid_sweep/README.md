# 11 -- exhaustive grid sweep

Every point on a Cartesian lattice.

What this shows:
- `LIB TADAH, ALGO GRID` -- TADAH's built-in grid sampler.
- `STEP <N>` -- samples per dimension. Total evals = `STEP ^ dim`,
  so this scales badly with dimensionality (`STEP=10` in 2D = 100
  evals; in 5D = 100 000).
- No `MAXEVAL` (the sweep runs to completion).

When to use: a quick deterministic survey of a low-dim search box,
or to seed a plot of the loss surface for inspection.

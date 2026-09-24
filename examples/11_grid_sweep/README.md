# 11 -- exhaustive grid sweep

Every point on a Cartesian lattice.

What this shows:
- `LIB TADAH, ALGO GRID` -- TADAH's built-in grid sampler.
- `PARAM GRID_POINTS <N>` -- samples per dimension. The sweep evaluates
  the INIT point (the config's values), then `GRID_POINTS ^ dim` grid
  nodes, so this scales badly with dimensionality (`GRID_POINTS=10` in 2D
  = 1 + 100 = 101 evals; in 5D = 100 001).
- No `MAXEVAL` (the sweep runs to completion). A `MAXEVAL` below the
  number of points stops the sweep after that many, in grid order (the
  INIT point, then the nodes, the first dimension varying fastest), and
  the journal says so (`GRID: MAXEVAL 5 < 101 grid points ...`).

When to use: a quick deterministic survey of a low-dim search box,
or to seed a plot of the loss surface for inspection.

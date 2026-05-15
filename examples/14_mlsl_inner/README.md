# 14 -- MLSL with INNER local optimiser

NLopt's MLSL needs a nested local descent.

What this shows:
- `STRATEGY { LIB NLOPT, ALGO G_MLSL_LDS, ... INNER { LIB ..., ALGO ... } }`
  -- the outer ALGO is the global driver, the `INNER` block configures
  the inner local optimiser the global driver delegates to.
- `LN_BOBYQA` is derivative-free (a Powell-style trust region), which
  matters here because finite-difference gradients on a noisy LAMMPS
  loss are unreliable.
- `POPULATION` caps the parallel-population size in algorithms that
  use one (MLSL keeps it as a per-basin uniqueness check).

The `INNER` block is mandatory for any AUGLAG- or MLSL-family algo;
the parser fails the config without it.

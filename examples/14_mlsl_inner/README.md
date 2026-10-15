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
  use one.

`G_MLSL_LDS` has no default local optimiser, so the `INNER` block is
required here: without it startup refuses the config, before any data is
loaded (`NLOPT:G_MLSL_LDS needs an INNER block (the STRATEGY-level
optimiser): NLopt has no default local optimiser for it; ...`). The same
holds for `G_MLSL`, `AUGLAG` and `AUGLAG_EQ`. The other MLSL and AUGLAG
forms run without one: `GN_MLSL`, `GN_MLSL_LDS`, `LN_AUGLAG` and
`LN_AUGLAG_EQ` use NLopt's `LN_COBYLA`, and `GD_MLSL`, `GD_MLSL_LDS`,
`LD_AUGLAG` and `LD_AUGLAG_EQ` use `LD_MMA`. An `INNER` block replaces
that default, and the local search alone decides whether the run takes a
gradient.

The `INNER` search is a local search: MLSL starts it at the points it
samples, so its `ALGO` is a local NLopt algorithm (`LN_BOBYQA` here). A
global one -- `GN_CRS2_LM`, a `GN_DIRECT` form, `GD_STOGO`, an MLSL form,
any NLopt algorithm whose name starts with `G` -- would search the whole
box from each of those points, and startup refuses it at the `ENDINNER`
line (`the INNER block of the STRATEGY-level optimiser: NLOPT:G_MLSL_LDS
cannot run NLOPT:GN_CRS2_LM as its INNER search: an INNER search is a
local search, ...`). Run a global search on its own instead: as a `PLAIN`
run or as the `PRESTAGE` exploration.

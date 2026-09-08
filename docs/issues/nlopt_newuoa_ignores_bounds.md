# LN_NEWUOA ignores the OPTIM box entirely

**Where:** nlopt 2.10 `src/api/optimize.c:849` calls `newuoa(..., lb=NULL, ub=NULL, ...)`; only
LN_NEWUOA_BOUND clamps interpolation points. HPO's registry flags it (`OptimizerCaps::ignores_bounds`) and
`scale_validation` prints a startup WARNING (phase 0).

**Risk:** evaluations outside the box reach the bump-support gate / FAILSCORE paths, waste budget and
can produce out-of-box "best" points that a later `INIT METHOD WARM` clamps.

**Proposal:** either unregister LN_NEWUOA from the caps registry (hard error naming LN_NEWUOA_BOUND /
LN_BOBYQA) or clamp inside the objective wrapper with a graded FAILSCORE band for out-of-box points.
Prefer unregistering: NLopt's own docs recommend BOBYQA over NEWUOA.

**Test:** config with `ALGO LN_NEWUOA` is rejected at startup (or every host() point lies inside the box).

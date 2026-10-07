# LN_NEWUOA ignores the OPTIM box entirely

**Where:** nlopt 2.10 `src/api/optimize.c:849` calls `newuoa(..., lb=NULL, ub=NULL, ...)`; only
LN_NEWUOA_BOUND clamps interpolation points, and HPO refuses LN_NEWUOA_BOUND since 1.6.0 (its nested MMA
subproblems stall for 10-60 s once its model is not finite: `nlopt_newuoa_bound_stall.md`). HPO's registry
flags LN_NEWUOA (`OptimizerCaps::ignores_bounds`) and `scale_validation` prints a startup WARNING (phase 0).

**Risk:** evaluations outside the box reach the bump-support gate / FAILSCORE paths, waste budget and
can produce out-of-box "best" points that a later `INIT METHOD WARM` clamps.

**Proposal:** either refuse LN_NEWUOA too (an entry of the caps registry's refusal table,
`optimizer_caps.cpp`, naming LN_BOBYQA, the one bound-constrained replacement) or clamp inside the objective
wrapper with a graded FAILSCORE band for out-of-box points. Prefer refusing: NLopt's own docs recommend
BOBYQA over NEWUOA. Also seen in the LN_NEWUOA_BOUND investigation: in 10 of 720 survey cases without
`XTOL` (a noisy sphere, n = 2) LN_NEWUOA passed NaN coordinates to the objective, and NEWUOA needs n >= 2
(NLopt returns `INVALID_ARGS` otherwise), which HPO does not check at startup.

**Test:** config with `ALGO LN_NEWUOA` is rejected at startup (or every host() point lies inside the box).

# LN_NEWUOA ignores the OPTIM box entirely (refused)

**Status:** refused since 2026-10-12 (owner decision taken in MK's absence: the `OPTIM` box is the
configuration's contract). `NLOPT:LN_NEWUOA` is an entry of the caps registry's refusal table
(`src/optimizer_caps.cpp`, `backend_refusal()`), as `LN_NEWUOA_BOUND` is: `tadah hpo` stops before it loads
any data, at the line that closes the block that names it (`ENDOPTIMIZER`, `ENDEXPLORE`, `ENDINNER`),
naming the phase and `NLOPT:LN_BOBYQA`, in 1-D and 2-D alike; no list of alternatives offers it and the
ALGO help (`tadah explain NLOPT:LN_NEWUOA`) prints the refusal. Tests: `test_optimizer_caps_step.cpp`,
`test_startup_validation_strategy.cpp` (`NLOPT:LN_NEWUOA, the Ceres least-squares solvers and an unknown
optimiser are refused ...`).

**Where:** nlopt 2.10 `src/api/optimize.c:849` calls `newuoa(..., lb=NULL, ub=NULL, ...)`; only
LN_NEWUOA_BOUND clamps interpolation points, and HPO refuses LN_NEWUOA_BOUND since 1.6.0 (its nested MMA
subproblems stall for 10-60 s once its model is not finite: `nlopt_newuoa_bound_stall.md`). Until the
refusal HPO's registry flagged LN_NEWUOA (`OptimizerCaps::ignores_bounds`) and `scale_validation` printed a
startup WARNING.

**What it did:** evaluations outside the box reached the bump-support gate / FAILSCORE paths, wasted budget
and produced out-of-box "best" points: a 2-D run (`RCUT2B` in [4, 6], `CGRID2B` in [0.1, 2], `MAXEVAL 25`)
made 20 of its 25 evaluations outside the box, and its best, `RCUT2B` 8.11, `CGRID2B` 2.03, became
`best_pot.tadah`. In 10 of 720 survey cases without `XTOL` (a noisy sphere, n = 2) LN_NEWUOA passed NaN
coordinates to the objective, and NEWUOA needs n >= 2: a 1-D run passed startup and ended `FAILED --
dimension 1 must be >= 2` after 0 evaluations (NLopt's `INVALID_ARGS` at run time).

**Replacement:** `NLOPT:LN_BOBYQA`, Powell's bound-constrained successor of NEWUOA (NLopt's own docs recommend
BOBYQA over NEWUOA). A study moved to it takes another trajectory: its `STEP` rescales each dimension
instead of setting one radius.

# 15 -- simulated annealing

Classic Kirkpatrick/Metropolis SA, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL` -- the built-in SA driver. No external
  dependency; uses HPO's own RNG.
- `PARAM` knobs tune the schedule:
  - `RT` -- geometric cooling factor.
  - `L` -- Markov-chain length per temperature step.
  - `STEP_FRAC` -- per-dim Gaussian step as a fraction of `high - low`;
    valid range `(0, 1]`.
  - `BOUNDARY_MODE` -- `0` (default) uses exact mirror reflection; `1`
    redraws each offending coordinate until it is in bounds, with a
    4096-draw guard that turns a broken proposal kernel into a hard error.
  - `T0_PROBES`, `T0_PACCEPT` -- T0 auto-calibration from full-vector Gaussian probes around the seed.
    It uses the median uphill finite delta, or the median absolute finite
    delta when none is uphill; `T0_PACCEPT` applies to that scale, and
    FAILSCORE probes are excluded.
- Explicit alternatives are fail-fast rather than last-one-wins: choose
  `STEP` or an explicit `STEP_FRAC`; choose `T_HOT` or explicitly tune
  `T0_PROBES`/`T0_PACCEPT`; and omit `REANNEAL_SCALE` when setting
  `REANNEAL 0`.
- Stop conditions honoured: `MAXEVAL` (hard cap including probes),
  `MAXTIME`, `STOPVAL`, `T_MIN` (auto = T0e-6; with `REANNEAL` the
  run warm-restarts from the best point only while the next scaled start can
  execute two rungs before falling below `T_MIN`), and
  `FTOL_REL`/`FTOL_ABS` via the Corana eps-NEPS plateau rule.
- Journal OOB percentages use `oob / proposed`; boundary correction does
  not create an additional host evaluation.
- The selected INIT seed must have exactly one finite, in-bounds value per
  transformed OPTIM dimension (positive before a log transform). NaN,
  infinity, and FAILSCORE proposals are rejected without poisoning a
  healthy chain.

Use when you want a simple, well-understood global optimiser with
explicit cooling control. See example 20 for the Corana adaptive
variant, 18 for parallel tempering, 19 for population annealing.

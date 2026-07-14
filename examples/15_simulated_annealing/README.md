# 15 -- simulated annealing

Classic Kirkpatrick/Metropolis SA, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL` -- the built-in SA driver. No external
  dependency; uses HPO's own RNG.
- `PARAM` knobs tune the schedule:
  - `RT` -- geometric cooling factor.
  - `L` -- Markov-chain length per temperature step.
  - `STEP_FRAC` -- per-dim Gaussian step as fraction of `high - low`.
  - `T0_PROBES`, `T0_PACCEPT` -- T0 auto-calibration from local probes
    around the seed point (target initial acceptance probability;
    FAILSCORE probes are excluded from the statistics).
- Stop conditions honoured: `MAXEVAL` (hard cap including probes),
  `MAXTIME`, `STOPVAL`, `T_MIN` (auto = T0e-6; with `REANNEAL` the
  run warm-restarts from the best point instead of dying there), and
  `FTOL_REL`/`FTOL_ABS` via the Corana eps-NEPS plateau rule.

Use when you want a simple, well-understood global optimiser with
explicit cooling control. See example 20 for the Corana adaptive
variant, 18 for parallel tempering, 19 for population annealing.

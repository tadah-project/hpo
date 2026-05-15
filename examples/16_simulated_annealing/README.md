# 16 -- simulated annealing

Classic Kirkpatrick/Metropolis SA, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL` -- the built-in SA driver. No external
  dependency; uses HPO's own RNG.
- `PARAM` knobs tune the schedule:
  - `RT` -- geometric cooling factor.
  - `L` -- Markov-chain length per temperature step.
  - `STEP_FRAC` -- per-dim Gaussian step as fraction of `high - low`.
  - `T0_PROBES`, `T0_PACCEPT` -- T0 auto-calibration from a short
    random walk (target initial acceptance probability).
- Stop conditions honoured: `MAXEVAL` (hard cap including probes),
  `MAXTIME`, `STOPVAL`. `FTOL_REL`/`FTOL_ABS` are not used.

Use when you want a simple, well-understood global optimiser with
explicit cooling control.

# 19 -- population annealing (ANNEAL_PA)

Population-based simulated annealing, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL_PA` -- a population of `POPULATION` walkers
  anneals from `T_HOT` down to `T_MIN`. Every temperature step:
  reweight by `exp(-dbeta * loss)`, resample (systematic; low-loss
  walkers multiply, FAILSCORE walkers die -- the natural cliff
  handling), then `SWEEPS` rungs of the stepper per walker.
- `PARAM ESS_TARGET 0.5` -- adaptive schedule: the temperature drop is
  chosen so the effective sample size stays at half the (finite)
  population; `RT` caps the per-step cooling rate. `ESS_TARGET 0`
  falls back to the pure geometric `T *= RT` schedule.
- Walker state `(x, f, V)` is copied on resample -- children inherit
  the parent's step calibration; each SLOT keeps its own RNG stream,
  so results are identical for any THREADS.
- The journal carries one line per temperature step: T, ESS, unique
  parents, culled walkers, finite fraction, population best.

Run:

    bash ../common/run.sh

# 19 -- population annealing (ANNEAL_PA)

Population-based simulated annealing, TADAH built-in.

What this shows:
- `LIB TADAH, ALGO ANNEAL_PA` -- a population of `POPULATION` walkers
  anneals from `T_HOT` down to `T_MIN`. Every temperature step:
  reweight by `exp(-dbeta * loss)`, resample (systematic; low-loss
  walkers multiply and non-finite/failure-band walkers receive zero weight
  when at least one feasible walker exists), then `SWEEPS` rungs of the
  stepper per walker.
  If every walker fails, resampling is skipped and geometric cooling
  continues for that step.
- `PARAM ESS_TARGET 0.5` -- adaptive schedule: the temperature drop is
  chosen so the effective sample size targets half the feasible
  population; `RT` caps the per-step cooling rate. `ESS_TARGET 0`
  falls back to the pure geometric `T *= RT` schedule.
- The adaptive solve is scale-aware. If the requested temperature rounds
  back to the current double, PA advances by one representable step and
  recomputes `dbeta` from the actual endpoint; weights and ESS use that same
  increment. Geometric cooling also progresses through subnormal values,
  and zero-weight parents are never selected by systematic resampling.
- Walker state `(x, f, V)` is copied on resample -- children inherit
  the parent's step calibration; each SLOT keeps its own RNG stream. Because
  this example fixes `POPULATION`, uses a deterministic objective, and has
  no timing-dependent stop, the same `SEED` gives the same evaluated set for any `THREADS`.
- The journal carries one line per temperature step: T, ESS, unique
  parents, culled lineages (`POPULATION - unique parents`), the legacy-named
  `finite` field (feasible fraction), and population best. `RESET_TO_BEST` is
  invalid for PA because resampling is the population-memory mechanism.

Run:

    bash ../common/run.sh

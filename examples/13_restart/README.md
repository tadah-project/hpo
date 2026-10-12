# 13 -- RESTART decorator

Auto-recover when the inner optimiser stalls.

What this shows:
- `STRATEGY { ... RESTART { ... } }` -- decorator that wraps the
  outer strategy. Knobs:
  - `AFTER_STAGNATION <N>` -- restart after N evals without improvement.
  - `MAX <K>` -- give up after K restarts.
  - `STRATEGY <init-strategy>` -- how to seed each restart
    (`RANDOM`, `LHS`, `SOBOL`); `STRATEGY` is the alias of the canonical
    `METHOD` keyword.
  - `SEED <n>` (not set here) -- restart r draws its point with
    `SEED + r`, so the restart points repeat from run to run; without
    it each restart draws from `std::random_device`.

RESTART relaunches a search that stagnates -- one that goes on
evaluating without a new best -- not one that ends: here DLIB:BFGS
reaches the upper bound `RCUT2B` = 6, takes its gradient there and ends
on its objective-delta test after 9 evaluations, before 3 pass without
a new best, so no restart fires (`DLIB:BFGS: FTOL_ABS reached`). Until
2026-10-12 BFGS evaluated the bound again at eval 10, which fired the
window, and two restarts followed. Useful for unattended overnight runs
where a search that keeps evaluating without progress (a stochastic or
global optimiser on a plateau) shouldn't waste the whole job.

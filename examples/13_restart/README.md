# 13 -- RESTART decorator

Auto-recover when the inner optimiser stalls.

What this shows:
- `STRATEGY { ... RESTART { ... } }` -- decorator that wraps the
  outer strategy. Knobs:
  - `AFTER_STAGNATION <N>` -- restart after N evals without improvement.
  - `MAX <K>` -- give up after K restarts: the next stagnation ends the
    run.
  - `METHOD <init-strategy>` -- how to seed each restart
    (`RANDOM`, `LHS`, `SOBOL`; `STRATEGY` is an alias).
  - `SEED <n>` -- restart r draws its point with `SEED + r`, so the
    restart points repeat from run to run (here `SEED 13`: restarts 1
    and 2 draw with 14 and 15). Without it the run draws one seed from
    `std::random_device` at its first restart, and the journal names it
    (`SEED <S>` in the RESTART block draws the same points again).

RESTART relaunches a search that stagnates -- one that goes on
evaluating without a new best -- not one that ends. Here NLOPT:LN_SBPLX
climbs to the upper bound `RCUT2B` = 6 and stays there: after 3
evaluations without a new best the window fires, and RESTART relaunches
it from a RANDOM point (`RCUT2B` = 4.541125, then 5.422967). Each round
climbs to the bound again and stagnates; the third stagnation ends the
run after 15 evaluations, `STOPPED -- stagnation: no new best in 3 evals
after 2 restart(s)`, with the best at `RCUT2B` = 6. The run repeats bit
for bit. (Until 2026-10-12 the example ran DLIB:BFGS without a SEED,
so its restart points came from `std::random_device` and differed from
run to run; with the repeat points of 2026-10-12 that search converged
before the window passed, and no restart fired.) Useful for unattended
overnight runs where a search that keeps evaluating without progress (a
local search held at a bound, a stochastic or global optimiser on a
plateau) shouldn't waste the whole job.

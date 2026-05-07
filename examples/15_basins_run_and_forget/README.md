# 15 — BASINS, the "run and forget" strategy

`STRATEGY { TYPE BASINS }` is an anytime/interruptible HPO policy designed
for hour/day/week-long invocations on workstations or SLURM clusters.

## What it does

Round k = 1, 2, 3, ... (unbounded by default):

1. **EXPLORE stage** at `EXPLORE.FTOL_LADDER[k]`. Runs the configured
   global optimiser (default `NLOPT:GN_CRS2_LM`) seeded from the basin
   pool's best basin (or from your `INIT { STRATEGY ... }` chain on
   round 1). Candidate basins discovered during this stage are harvested
   into the persistent **basin pool** with deduplication.
2. **REFINE stage** at the cumulative ladder
   `REFINE.FTOL_LADDER[1..k]`. Round 1 visits one (loose) rung; round 5
   visits all five rungs cumulatively, refining every basin in the
   pool that has not yet been resolved at that tolerance.

After every stage the policy commits a checkpoint to
`checkpoint.tadah` in the working directory. SIGINT (ctrl-C),
SIGTERM (`kill`, SLURM cancellation), and SIGUSR2 (the SLURM advance
warning configured via `--signal=B:USR2@<seconds>`) all trigger a
clean shutdown after the current evaluation: a final checkpoint is
written, the journal is closed, and the process exits.

Re-running the same command in the same directory reads
`checkpoint.tadah`, restores the basin pool + journal counters, and
picks up at the saved (round, stage). Optimiser internal state is
*not* required — the per-basin best-known params drive the resume.

## Defaults

The minimal config below is functional:

```
OPTIMIZER
  STRATEGY
    TYPE BASINS
    MAXTIME 3600
  ENDSTRATEGY
ENDOPTIMIZER
```

Which expands to (defaults applied):
- `EXPLORE`: `NLOPT:GN_CRS2_LM` (population-based, noise-tolerant).
- `REFINE`:  `NLOPT:LN_BOBYQA` (derivative-free quadratic trust-region,
  bound-aware). LN_BOBYQA replaced an earlier LN_SBPLX default — SBPLX
  silently collapses ("simplex too small in dimension X") near a bound
  and exits with masked `XTOL_REACHED`, ending refinement without progress.
- 6-rung default ladder starting at `1e-3`, descending by 0.1× per round,
  floored at `max(1e-9, 10·noise_rel_spread)` from `NOISE.CALIBRATION`.
- 32-basin pool, 5% relative dedup distance, 5000 evals/stage cap.
- 5 consecutive no-improvement rounds → STOP_NO_IMPROVE_ROUNDS stop.

## Opt-in tuning knobs (inside the STRATEGY block)

`SEED_TOP_N <N>` — at the start of each round k>1 EXPLORE stage,
pre-evaluate the top-N pool basins via `host()` (the per-eval logger
captures them) and seed the optimiser from the lowest-loss of them.
Default `1` preserves the existing behaviour (only the best basin
seeds). Useful when the pool has multiple comparable-loss basins and
you want the round-2+ EXPLORE stage to consider all of them as starting
points before settling on one.

`CLAMP_TO_NOISE_FLOOR true|false` — strictly trim each stage's
`FTOL_LADDER` rungs below the `suggested_ftol` reported by
`NOISE.CALIBRATION`. Default `false`. Opt in once the seed produces a
converged-enough potential that `NOISE.CALIBRATION`'s `rel_spread`
reflects real objective noise — at a poor seed (where the BAD_POT
clamp at `TADAH_HPO_BAD_POT_ERMSE_MEV` fires for some DBFILEs) the
`rel_spread` is misleading and the clamp would prune too aggressively.
The policy emits a warning if the clamp drops more than half of either
ladder, so you'll see the unsafe case in `hpo_run.log`.

## What basins refinement actually finds

A basin's "loss" stored in the pool is the minimum found by `host()`
during its REFINE stage, not just the EXPLORE-discovery sample.
A basin discovered at loss=900 in EXPLORE can refine to loss=37 in
REFINE — the REFINE stage's bound-aware optimiser routinely escapes
the dedup-neighborhood (~5% relative L2) of its starting point and
finds a different local minimum elsewhere in the bbox. Don't try
to prune basins by their initial loss: the high-loss ones are often
the ones that produce the best refined points.

## What to look for

In `hpo_run.log` (`tail -f` for a live monitor):

```
[14:23:45]         BASINS round=1 EXPLORE begin: NLOPT:GN_CRS2_LM ftol_rel=1.000e-03 maxeval=400
[14:24:30]  BEST   eval=312 loss=8.91e-3 (...)
[14:25:11]         BASINS round=1 EXPLORE end: harvested 17 candidate(s); pool now 9 basin(s)
[14:25:11]         BASINS round=1 LOCAL step=1 ftol_rel=1.000e-03  basins=9/9
[14:25:48]         BASINS basin[0] tol=1.000e-03  loss=8.71e-3 (improved)
[14:26:31]         BASINS basin[1] tol=1.000e-03  loss=8.92e-3 (no improvement)
[14:30:02]         BASINS checkpoint saved: checkpoint.tadah
...
[15:00:08]         BASINS round=2 LOCAL step=2 ftol_rel=1.000e-04  basins=6/12
```

`BEST` events become rarer as rounds progress — that's the
expected anytime profile (concave-monotone improvement curve).

## SLURM usage

```bash
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR2@45    # SIGUSR2 45 s before the wall-clock kill
tadah hpo
```

The 45 s warning is enough for a checkpoint commit even when the
current evaluation is a 30 s LAMMPS run. After the job finishes,
re-`sbatch` the same script (or use `--requeue`) and the next
allocation resumes the basin pool at the saved (round, stage).

## When NOT to use BASINS

- For short, bounded runs (`MAXEVAL` < ~1000) where a single global
  search converges quickly — example 13 (`GN_CRS2_LM`) or example 1
  (`DLIB:BFGS`) is simpler.
- When you want a deterministic two-phase global → local handoff
  (example 7, `STRATEGY { TYPE PRESTAGE }`).
- When you want explicit per-start budget control (example 3,
  `STRATEGY { TYPE MULTISTART }`).

## Files this example produces

Standard HPO output (`loss.tadah`, `params.tadah`, `best_*.tadah`,
`hpo_run.log`) **plus**:

- `checkpoint.tadah` — atomically-committed key/value file with the
  basin pool, current round/stage, and journal counters. Inspectable
  with `cat`. Removed cleanly only if BASINS finishes naturally
  (e.g. STOP_NO_IMPROVE_ROUNDS stop). Keep around between SLURM jobs.

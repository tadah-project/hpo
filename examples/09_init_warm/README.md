# 09 -- warm restart

Continue from a previous run's optimum.

What this shows:
- `STRATEGY WARM <file>` -- load the OPTIM values and the per-script
  warmstart cache from `<file>` (a previous `best_pot.tadah`).
- `WARM_ON_MISMATCH warn` -- if the search-space layout in the file
  differs from the current config (e.g. extra OPTIM dim), log a
  warning and fall back to the bounds midpoint instead of aborting.

The local `run.sh` bootstraps a `seed_pot.tadah` on first invocation
by running a quick DLIB:BFGS pass, then hands off to the shared
driver. Subsequent runs reuse the cached seed.

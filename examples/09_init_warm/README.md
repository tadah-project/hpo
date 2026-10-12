# 09 -- warm restart

Continue from a previous run's optimum.

What this shows:
- `STRATEGY WARM <file>` -- load the OPTIM values and the per-script
  warmstart cache from `<file>` (a previous `best_pot.tadah`).
- `WARM_ON_MISMATCH warn` -- if the file does not give every OPTIM
  value (e.g. an extra OPTIM dim), or gives one outside its box, log a
  warning and keep the point instead of aborting: a missing value
  takes its value in the config (`-c`), and a value outside the box is
  clamped into it (`error` stops the run, `skip` drops the point).

The local `run.sh` bootstraps a `seed_pot.tadah` on first invocation
by running a quick DLIB:BFGS pass, then hands off to the shared
driver. Subsequent runs reuse the cached seed.
